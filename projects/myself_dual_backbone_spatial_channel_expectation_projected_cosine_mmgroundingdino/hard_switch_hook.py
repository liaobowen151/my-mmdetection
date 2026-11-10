"""Epoch schedule for delayed projected-cosine E1-Z activation."""

import torch

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class ProjectedCosineE1ZScheduleHook(Hook):
    """Run exact base fusion first, then enable projected-cosine E1-Z."""

    priority = 'VERY_HIGH'

    def __init__(self, start_epoch: int = 11) -> None:
        if start_epoch <= 1:
            raise ValueError('start_epoch must be greater than 1.')
        self.start_epoch = int(start_epoch)
        self._frozen_values = None
        self._frozen_parameters = None

    @staticmethod
    def _fusion(runner):
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        fusion = model.multimodal_fusion
        required = ('set_projected_cosine_e1z_enabled',
                    'is_projected_cosine_e1z_enabled')
        if any(not hasattr(fusion, name) for name in required):
            raise TypeError(
                'Fusion module does not support projected-cosine switching.')
        return fusion

    def before_train(self, runner) -> None:
        """Snapshot new fusion parameters after checkpoint loading."""
        fusion = self._fusion(runner)
        self._frozen_parameters = list(fusion.parameters())
        self._frozen_values = [
            parameter.detach().clone()
            for parameter in self._frozen_parameters
        ]

    def before_train_epoch(self, runner) -> None:
        fusion = self._fusion(runner)
        epoch = runner.epoch + 1
        enabled = epoch >= self.start_epoch
        was_enabled = fusion.is_projected_cosine_e1z_enabled()
        fusion.set_projected_cosine_e1z_enabled(enabled)
        if get_rank() == 0 and enabled != was_enabled:
            runner.logger.info(
                'Hard-switch fusion: projected-cosine E1-Z %s at epoch %d.',
                'enabled' if enabled else 'disabled', epoch)

    def after_train_iter(self,
                         runner,
                         batch_idx: int,
                         data_batch=None,
                         outputs=None) -> None:
        """Undo AdamW updates during epochs 1--10, including its state."""
        del batch_idx, data_batch, outputs
        if runner.epoch + 1 >= self.start_epoch:
            return
        if self._frozen_parameters is None or self._frozen_values is None:
            raise RuntimeError('Fusion parameter snapshot was not created.')

        optimizer = runner.optim_wrapper.optimizer
        with torch.no_grad():
            for parameter, frozen_value in zip(
                    self._frozen_parameters, self._frozen_values):
                parameter.copy_(frozen_value)
                # Start projected-cosine optimization from a clean AdamW
                # state when it becomes active at start_epoch.
                optimizer.state.pop(parameter, None)
