# Copyright (c) OpenMMLab. All rights reserved.
from numbers import Number
from typing import Optional

import torch
from mmengine.dist import all_reduce, get_rank
from mmengine.hooks import Hook
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class EpochLossAverageHook(Hook):
    """Log the distributed mean training loss at the end of every epoch.

    The hook averages the total loss returned by ``model.train_step`` over all
    iterations and all distributed ranks. Only rank 0 writes the result to the
    text log and visualizer.

    Args:
        loss_key (str): Key of the total loss in the training outputs.
            Defaults to ``'loss'``.
        log_name (str): Scalar name used by visualization backends.
            Defaults to ``'train/epoch_avg_loss'``.
    """

    priority = 'LOW'

    def __init__(self,
                 loss_key: str = 'loss',
                 log_name: str = 'train/epoch_avg_loss') -> None:
        if not isinstance(loss_key, str) or not loss_key:
            raise ValueError('loss_key must be a non-empty string.')
        if not isinstance(log_name, str) or not log_name:
            raise ValueError('log_name must be a non-empty string.')
        self.loss_key = loss_key
        self.log_name = log_name
        self._loss_sum = 0.0
        self._num_iters = 0

    def before_train_epoch(self, runner: Runner) -> None:
        """Reset statistics before the first iteration of each epoch."""
        self._loss_sum = 0.0
        self._num_iters = 0

    def after_train_iter(self,
                         runner: Runner,
                         batch_idx: int,
                         data_batch: Optional[dict] = None,
                         outputs: Optional[dict] = None) -> None:
        """Accumulate the detached total loss of the current iteration."""
        if outputs is None or self.loss_key not in outputs:
            raise KeyError(
                f'EpochLossAverageHook expected outputs[{self.loss_key!r}] '
                'after every training iteration.')

        loss = outputs[self.loss_key]
        if isinstance(loss, torch.Tensor):
            if loss.numel() != 1:
                loss = loss.mean()
            loss_value = loss.detach().item()
        elif isinstance(loss, Number):
            loss_value = float(loss)
        else:
            raise TypeError(
                f'outputs[{self.loss_key!r}] must be a scalar tensor or '
                f'number, but got {type(loss)}.')

        self._loss_sum += float(loss_value)
        self._num_iters += 1

    def after_train_epoch(self, runner: Runner) -> None:
        """Synchronize and log the mean loss over the completed epoch."""
        statistics = torch.tensor(
            [self._loss_sum, float(self._num_iters)], dtype=torch.float64)
        all_reduce(statistics, op='sum')

        global_num_iters = int(statistics[1].item())
        if global_num_iters == 0:
            if get_rank() == 0:
                runner.logger.warning(
                    'EpochLossAverageHook received no training losses in '
                    'epoch %d.', runner.epoch + 1)
            return

        epoch_avg_loss = statistics[0].item() / global_num_iters
        if get_rank() == 0:
            epoch = runner.epoch + 1
            runner.logger.info(
                'Epoch(train) [%d/%d]  epoch_avg_loss: %.6f', epoch,
                runner.max_epochs, epoch_avg_loss)
            runner.visualizer.add_scalar(
                self.log_name, epoch_avg_loss, step=epoch)

