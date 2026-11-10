"""Select unique M3FD test images stratified by GT size and parse subjects."""

import argparse
import json
from collections import defaultdict
from pathlib import Path

import spacy
from transformers import AutoTokenizer


def subject_head(doc):
    """Return the syntactic subject head, falling back for caption fragments."""
    subjects = [token for token in doc if token.dep_ in ('nsubj', 'nsubjpass')
                and token.pos_ in ('NOUN', 'PROPN', 'PRON')]
    if subjects:
        return subjects[0], 'nsubj'
    nouns = [token for token in doc if token.pos_ in ('NOUN', 'PROPN')]
    if nouns:
        return nouns[0], 'first_noun'
    return None, 'unresolved'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--ann-file', required=True)
    parser.add_argument('--bert-dir', required=True)
    parser.add_argument('--num-samples', type=int, default=50)
    parser.add_argument('--output', required=True)
    parser.add_argument('--review-output',
                        help='Optional human-review Markdown table.')
    args = parser.parse_args()

    nlp = spacy.load('en_core_web_sm', disable=['ner'])
    tokenizer = AutoTokenizer.from_pretrained(args.bert_dir, local_files_only=True)
    annotation = json.loads(Path(args.ann_file).read_text())
    annotations = {item['image_id']: item for item in annotation['annotations']}
    eligible = []
    for index, image in enumerate(annotation['images']):
        caption = image['caption']
        # Synthetic captions occasionally explicitly state that the referent
        # is absent; they cannot support a subject-token alignment audit.
        if caption.lstrip().lower().startswith('no '):
            continue
        doc = nlp(caption)
        head, method = subject_head(doc)
        if head is None:
            continue
        subject = head.text.lower()
        bert_tokens = tokenizer.tokenize(caption)
        if subject not in bert_tokens:
            continue
        ann = annotations[image['id']]
        bbox = ann['bbox']
        area_ratio = bbox[2] * bbox[3] / (image['width'] * image['height'])
        eligible.append(dict(
            sample_index=index, image_id=image['id'], file_name=image['file_name'],
            caption=caption, subject=subject, subject_lemma=head.lemma_.lower(),
            subject_dependency=method, subject_pos=head.pos_,
            bert_tokens=bert_tokens, bbox_xywh=bbox, area_ratio=area_ratio,
            scale_class=image.get('scale_cls')))

    # One referring expression per photograph. Prefer an explicit nsubj when
    # several eligible expressions share the same source image.
    by_file = defaultdict(list)
    for item in eligible:
        by_file[item['file_name']].append(item)
    candidates = []
    for values in by_file.values():
        values.sort(key=lambda item: (item['subject_dependency'] != 'nsubj',
                                      item['sample_index']))
        candidates.append(values[0])
    candidates.sort(key=lambda item: item['area_ratio'])

    selected = []
    used_files = set()
    for slot in range(args.num_samples):
        desired = round(slot * (len(candidates) - 1) / (args.num_samples - 1))
        for distance in range(len(candidates)):
            choices = (desired + distance, desired - distance)
            chosen = None
            for candidate_index in choices:
                if 0 <= candidate_index < len(candidates):
                    candidate = candidates[candidate_index]
                    if candidate['file_name'] not in used_files:
                        chosen = candidate
                        break
            if chosen is not None:
                selected.append(chosen)
                used_files.add(chosen['file_name'])
                break

    output = dict(
        selection_method='unique images, evenly spaced GT-area quantiles',
        parser='en_core_web_sm',
        num_candidates=len(candidates),
        samples=selected)
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(output, ensure_ascii=False, indent=2))
    if args.review_output:
        review_path = Path(args.review_output)
        review_path.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            '# 50-sample subject review', '',
            '| # | Sample | Image | Area ratio | Subject | Method | Caption |',
            '|---:|---:|---|---:|---|---|---|'
        ]
        for number, item in enumerate(selected, start=1):
            caption = item['caption'].replace('|', '\\|')
            lines.append(
                f"| {number} | {item['sample_index']} | {item['file_name']} | "
                f"{item['area_ratio']:.6f} | `{item['subject']}` | "
                f"{item['subject_dependency']} | {caption} |")
        review_path.write_text('\n'.join(lines) + '\n')
    print('selected', len(selected), 'unique images')
    for item in selected:
        print(item['sample_index'], item['subject'], item['subject_dependency'],
              f"area={item['area_ratio']:.6f}", item['caption'])


if __name__ == '__main__':
    main()
