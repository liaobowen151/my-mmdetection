"""Benchmark reference and exact C++ cross-modal selectors."""

import argparse
import time

import torch

from verify_cpp_exact_cross_modal_selector import make_candidates
from projects.myself_multimodal_mmgroundingdino_balanced_selector.models.cross_modal_balanced_competitive_selector import (  # noqa: E501
    CrossModalBalancedCompetitiveSelector,
)
from projects.myself_multimodal_mmgroundingdino_balanced_selector_cpp_exact.models.cpp_exact_cross_modal_balanced_competitive_selector import (  # noqa: E501
    CppExactCrossModalBalancedCompetitiveSelector,
)


def measure(selector, candidates, iterations):
    for _ in range(10):
        selector.select_final_queries(candidates)
    torch.cuda.synchronize()
    start = time.perf_counter()
    for _ in range(iterations):
        selector.select_final_queries(candidates)
    torch.cuda.synchronize()
    return (time.perf_counter() - start) * 1000 / iterations


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--batch-size', type=int, default=2)
    parser.add_argument('--iterations', type=int, default=100)
    args = parser.parse_args()
    kwargs = dict(pre_topk_per_modality=600, final_topk=900,
                  selection_mode='cross_modal_greedy_nms',
                  cross_modal_iou_threshold=.8)
    reference = CrossModalBalancedCompetitiveSelector(**kwargs)
    exact = CppExactCrossModalBalancedCompetitiveSelector(**kwargs)
    candidates = make_candidates(args.batch_size, 600, 256, seed=42)
    expected = reference.select_final_queries(candidates)
    actual = exact.select_final_queries(candidates)
    if not torch.equal(expected, actual):
        raise AssertionError('Selector outputs differ; aborting benchmark.')
    reference_ms = measure(reference, candidates, args.iterations)
    exact_ms = measure(exact, candidates, args.iterations)
    print(f'reference: {reference_ms:.3f} ms/iteration')
    print(f'cpp-exact: {exact_ms:.3f} ms/iteration')
    print(f'speedup:   {reference_ms / exact_ms:.2f}x')


if __name__ == '__main__':
    main()
