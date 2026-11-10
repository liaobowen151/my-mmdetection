# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Dict, Optional

import torch
from mmengine.dist import all_reduce, get_rank
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class EpochIRGuidedMetricsHook(Hook):
    """Log epoch means for IR-guided encoder diagnostics and VIS cls loss."""

    priority = 'LOW'

    metric_map = {
        'ir_hungarian_pos_iou': 'epoch_ir_hungarian_pos_iou',
        'ir_topk_max_iou': 'epoch_ir_topk_max_iou',
        'ir_topk_recall_50': 'epoch_ir_topk_recall_50',
        'ir_topk_recall_70': 'epoch_ir_topk_recall_70',
        'ir_matching_iou_gap': 'epoch_ir_matching_iou_gap',
        'vis_enc_loss_cls': 'epoch_ir_vis_cls_loss',
        'vis_hungarian_pos_iou': 'epoch_vis_hungarian_pos_iou',
        'vis_selected_max_iou': 'epoch_vis_selected_max_iou',
        'ir_vis_pos_index_agreement': 'epoch_ir_vis_pos_index_agreement',
    }

    def before_train_epoch(self, runner: Runner) -> None:
        self._sums: Dict[str, float] = {
            key: 0.0 for key in self.metric_map
        }
        self._counts: Dict[str, int] = {
            key: 0 for key in self.metric_map
        }

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        if outputs is None:
            return
        for key in self.metric_map:
            if key not in outputs:
                continue
            value = outputs[key]
            if isinstance(value, torch.Tensor):
                value = value.detach().mean().item()
            elif isinstance(value, Number):
                value = float(value)
            else:
                continue
            self._sums[key] += value
            self._counts[key] += 1

    def after_train_epoch(self, runner: Runner) -> None:
        epoch = runner.epoch + 1
        reported = []
        for key, log_name in self.metric_map.items():
            statistics = torch.tensor(
                [self._sums[key], float(self._counts[key])],
                dtype=torch.float64)
            all_reduce(statistics, op='sum')
            count = statistics[1].item()
            if count == 0:
                continue
            mean = statistics[0].item() / count
            if get_rank() == 0:
                runner.visualizer.add_scalar(
                    'train/' + log_name, mean, step=epoch)
                reported.append(f'{log_name}: {mean:.6f}')
        if get_rank() == 0 and reported:
            runner.logger.info(
                'Epoch(train) [%d/%d]  %s', epoch, runner.max_epochs,
                '  '.join(reported))
