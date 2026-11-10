"""Verify the reference-IoU C++ selector against Python greedy NMS."""

import argparse

import torch

from projects.myself_multimodal_mmgroundingdino_balanced_selector.models.cross_modal_balanced_competitive_selector import (  # noqa: E501
    CrossModalBalancedCompetitiveSelector,
)
from projects.myself_multimodal_mmgroundingdino_balanced_selector_cpp_exact.models.cpp_exact_cross_modal_balanced_competitive_selector import (  # noqa: E501
    CppExactCrossModalBalancedCompetitiveSelector,
)


def make_candidates(batch_size, pre_topk, text_len, seed):
    generator = torch.Generator(device='cuda').manual_seed(seed)
    scores = torch.randn(batch_size, 2 * pre_topk, text_len,
                         generator=generator, device='cuda')
    coords = torch.randn(batch_size, 2 * pre_topk, 4,
                         generator=generator, device='cuda')
    coords[..., 2:] = coords[..., 2:].abs()
    coords[:, pre_topk:pre_topk + 64] = coords[:, :64]
    return dict(
        scores=scores,
        coords_unact=coords,
        features=torch.randn(batch_size, 2 * pre_topk, 256,
                             generator=generator, device='cuda'),
        source_indices=torch.arange(2 * pre_topk, device='cuda').expand(
            batch_size, -1),
        modality_ids=torch.cat([
            torch.zeros(batch_size, pre_topk, dtype=torch.long,
                        device='cuda'),
            torch.ones(batch_size, pre_topk, dtype=torch.long,
                       device='cuda'),
        ], dim=1))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--threshold', type=float, default=None)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required.')
    thresholds = (args.threshold, ) if args.threshold is not None else \
        (0., .8, 1.)
    for threshold in thresholds:
        reference = CrossModalBalancedCompetitiveSelector(
            pre_topk_per_modality=600, final_topk=900,
            selection_mode='cross_modal_greedy_nms',
            cross_modal_iou_threshold=threshold)
        exact = CppExactCrossModalBalancedCompetitiveSelector(
            pre_topk_per_modality=600, final_topk=900,
            selection_mode='cross_modal_greedy_nms',
            cross_modal_iou_threshold=threshold)
        for seed in range(args.trials):
            candidates = make_candidates(args.batch_size, 600, 256, seed)
            expected = reference.select_final_queries(candidates)
            actual = exact.select_final_queries(candidates)
            if not torch.equal(expected, actual):
                mismatch = (expected != actual).sum().item()
                raise AssertionError(
                    f'threshold={threshold}, seed={seed}: {mismatch} '
                    'selected indices differ.')
    print('PASS: exact C++ selector matches reference selected indices.')


if __name__ == '__main__':
    main()
