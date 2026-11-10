#!/usr/bin/env python3
import argparse
import json
import random
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.nn.utils import clip_grad_norm_
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import resolve_device, write_json
from model import SubjectHeadClassifier


class FeatureDataset(Dataset):
    def __init__(self, path: str):
        self.data = torch.load(path, map_location='cpu')
        self.records = self.data['records']

    def __len__(self):
        return len(self.records)

    def __getitem__(self, index):
        return self.records[index]


def collate(records):
    max_len = max(record['hidden'].shape[0] for record in records)
    dim = records[0]['hidden'].shape[-1]
    hidden = torch.zeros(len(records), max_len, dim, dtype=torch.float32)
    mask = torch.zeros(len(records), max_len, dtype=torch.bool)
    targets = torch.empty(len(records), dtype=torch.long)
    for i, record in enumerate(records):
        length = record['hidden'].shape[0]
        hidden[i, :length] = record['hidden'].float()
        mask[i, :length] = record['candidate_mask']
        targets[i] = record['target']
    return hidden, mask, targets, records


def evaluate(model, loader, device):
    model.eval()
    total = correct = top2_correct = 0
    loss_sum = 0.0
    predictions = []
    domain_total = Counter()
    domain_correct = Counter()
    with torch.inference_mode():
        for hidden, mask, targets, records in loader:
            hidden, mask, targets = hidden.to(device), mask.to(device), targets.to(device)
            logits = model(hidden, mask)
            loss = F.cross_entropy(logits, targets)
            probabilities = logits.softmax(dim=-1)
            predicted = logits.argmax(dim=-1)
            top2 = logits.topk(k=min(2, logits.shape[-1]), dim=-1).indices
            batch_size = targets.numel()
            total += batch_size
            correct += (predicted == targets).sum().item()
            top2_correct += (top2 == targets[:, None]).any(dim=1).sum().item()
            loss_sum += loss.item() * batch_size
            for row, record in enumerate(records):
                pred_index = predicted[row].item()
                left, right = record['offsets'][pred_index]
                predictions.append({
                    'sample_index': record['sample_index'],
                    'filename': record['filename'],
                    'caption': record['caption'],
                    'gold_head': record['subject_head'],
                    'predicted_head': record['caption'][left:right],
                    'gold_span': record['subject_head_char_span'],
                    'predicted_span': [left, right],
                    'confidence': probabilities[row, pred_index].item(),
                    'correct': pred_index == targets[row].item(),
                    'dataset_name': record.get('dataset_name', 'single'),
                })
                domain = record.get('dataset_name', 'single')
                domain_total[domain] += 1
                domain_correct[domain] += int(pred_index == targets[row].item())
    domain_accuracy = {
        domain: domain_correct[domain] / count
        for domain, count in sorted(domain_total.items())
    }
    return {
        'loss': loss_sum / max(total, 1),
        'exact_match': correct / max(total, 1),
        'top2_accuracy': top2_correct / max(total, 1),
        'num_samples': total,
        'domain_exact_match': domain_accuracy,
        'macro_exact_match': sum(domain_accuracy.values()) / max(len(domain_accuracy), 1),
    }, predictions


def parse_args():
    parser = argparse.ArgumentParser(description='Train only SubjectHeadClassifier.')
    parser.add_argument('--feature-dir', required=True)
    parser.add_argument('--output-dir', required=True)
    parser.add_argument('--hidden-dim', type=int, default=256)
    parser.add_argument('--dropout', type=float, default=0.1)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight-decay', type=float, default=1e-2)
    parser.add_argument('--batch-size', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=50)
    parser.add_argument('--patience', type=int, default=8)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--device', default='auto')
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument(
        '--sampling', choices=['natural', 'sqrt', 'equal'], default='natural',
        help='Dataset sampling. sqrt is recommended for combined training.')
    return parser.parse_args()


def main():
    args = parse_args()
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    feature_dir = Path(args.feature_dir)
    datasets = {
        split: FeatureDataset(str(feature_dir / f'{split}_features.pt'))
        for split in ('train', 'val', 'test')
    }
    generator = torch.Generator().manual_seed(args.seed)
    train_sampler = None
    if args.sampling != 'natural':
        domain_counts = Counter(
            record.get('dataset_name', 'single')
            for record in datasets['train'].records)
        alpha = 0.5 if args.sampling == 'sqrt' else 0.0
        weights = [
            domain_counts[record.get('dataset_name', 'single')] ** (alpha - 1.0)
            for record in datasets['train'].records
        ]
        train_sampler = WeightedRandomSampler(
            weights, num_samples=len(weights), replacement=True,
            generator=generator)
        print('train_domain_counts:', dict(domain_counts),
              'sampling:', args.sampling)
    loaders = {}
    for split, dataset in datasets.items():
        is_train = split == 'train'
        loaders[split] = DataLoader(
            dataset, batch_size=args.batch_size,
            shuffle=is_train and train_sampler is None,
            sampler=train_sampler if is_train else None,
            collate_fn=collate, num_workers=args.num_workers,
            generator=generator if is_train and train_sampler is None else None)
    input_dim = datasets['train'].records[0]['hidden'].shape[-1]
    device = resolve_device(args.device)
    model = SubjectHeadClassifier(input_dim, args.hidden_dim, args.dropout).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    history = []
    best_accuracy = -1.0
    best_overall_accuracy = -1.0
    best_loss = float('inf')
    epochs_without_improvement = 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_total = train_loss = train_correct = 0
        for hidden, mask, targets, _ in loaders['train']:
            hidden, mask, targets = hidden.to(device), mask.to(device), targets.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(hidden, mask)
            loss = F.cross_entropy(logits, targets)
            loss.backward()
            clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item() * targets.numel()
            train_correct += (logits.argmax(-1) == targets).sum().item()
            train_total += targets.numel()
        val_metrics, _ = evaluate(model, loaders['val'], device)
        row = {
            'epoch': epoch,
            'train_loss': train_loss / train_total,
            'train_exact_match': train_correct / train_total,
            **{f'val_{k}': v for k, v in val_metrics.items()},
        }
        history.append(row)
        print(json.dumps(row))
        selection_accuracy = val_metrics['macro_exact_match']
        improved = (
            selection_accuracy > best_accuracy
            or (selection_accuracy == best_accuracy
                and val_metrics['exact_match'] > best_overall_accuracy)
            or (selection_accuracy == best_accuracy
                and val_metrics['exact_match'] == best_overall_accuracy
                and val_metrics['loss'] < best_loss))
        if improved:
            best_accuracy = selection_accuracy
            best_overall_accuracy = val_metrics['exact_match']
            best_loss = val_metrics['loss']
            epochs_without_improvement = 0
            # Deliberately save classifier parameters only; never save BERT.
            torch.save(model.state_dict(), output_dir / 'subject_head_classifier_best.pth')
        else:
            epochs_without_improvement += 1
        torch.save(model.state_dict(), output_dir / 'subject_head_classifier_last.pth')
        write_json(str(output_dir / 'training_history.json'), history)
        if epochs_without_improvement >= args.patience:
            break

    state_dict = torch.load(
        output_dir / 'subject_head_classifier_best.pth', map_location=device)
    model.load_state_dict(state_dict)
    val_metrics, val_predictions = evaluate(model, loaders['val'], device)
    test_metrics, test_predictions = evaluate(model, loaders['test'], device)
    summary = {
        'model_config': {
            'input_dim': input_dim,
            'hidden_dim': args.hidden_dim,
            'dropout': args.dropout,
        },
        'checkpoint_contains_bert': False,
        'best_val': val_metrics,
        'test': test_metrics,
    }
    write_json(str(output_dir / 'metrics.json'), summary)
    write_json(str(output_dir / 'val_predictions.json'), val_predictions)
    write_json(str(output_dir / 'test_predictions.json'), test_predictions)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
