"""Epoch schedule and diagnostics for hard-switch fusion experiments."""

from mmengine.dist import get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class HardSwitchFusionScheduleHook(Hook):
    """Enable E1-Z at ``start_epoch`` and log cross-attention parameter norms."""

    priority = 'VERY_HIGH'

    def __init__(self, start_epoch: int = 11) -> None:
        if start_epoch <= 1:
            raise ValueError('start_epoch must be greater than 1.')
        self.start_epoch = int(start_epoch)

    @staticmethod
    def _fusion(runner):
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        fusion = model.multimodal_fusion
        required = ('set_spatial_expectation_enabled',
                    'is_spatial_expectation_enabled', 'cross_modal_attention')
        if any(not hasattr(fusion, name) for name in required):
            raise TypeError('Model fusion does not support hard-switch training.')
        return fusion

    def before_train_epoch(self, runner) -> None:
        fusion = self._fusion(runner)
        enabled = runner.epoch + 1 >= self.start_epoch
        was_enabled = fusion.is_spatial_expectation_enabled()
        fusion.set_spatial_expectation_enabled(enabled)
        if get_rank() == 0 and enabled != was_enabled:
            runner.logger.info(
                'Hard-switch fusion: E1-Z spatial expectation %s at epoch %d.',
                'enabled' if enabled else 'disabled', runner.epoch + 1)

    def after_train_epoch(self, runner) -> None:
        fusion = self._fusion(runner)
        if get_rank() != 0:
            return
        stage = float(fusion.is_spatial_expectation_enabled())
        cross = fusion.cross_modal_attention
        diagnostics = {
            'stage/spatial_expectation_enabled': stage,
            'cross/vis_from_ir_value_proj_weight_norm':
                cross.vis_from_ir.value_proj.weight.detach().norm().item(),
            'cross/ir_from_vis_value_proj_weight_norm':
                cross.ir_from_vis.value_proj.weight.detach().norm().item(),
            'cross/vis_from_ir_output_proj_weight_norm':
                cross.vis_from_ir.output_proj.weight.detach().norm().item(),
            'cross/ir_from_vis_output_proj_weight_norm':
                cross.ir_from_vis.output_proj.weight.detach().norm().item(),
        }
        step = runner.epoch + 1
        for key, value in diagnostics.items():
            runner.logger.info('Epoch(train) [%d/%d]  fusion/%s: %.6f',
                               step, runner.max_epochs, key, value)
            runner.visualizer.add_scalar(f'train/fusion/{key}', value,
                                         step=step)
