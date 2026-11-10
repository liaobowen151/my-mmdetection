"""Epoch statistics for spatial/channel expectation fusion."""

from typing import Dict, Optional

from mmengine.dist import all_reduce, get_rank
from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.runner import Runner

from mmdet.registry import HOOKS


@HOOKS.register_module()
class SpatialChannelExpectationStatisticsHook(Hook):
    """Aggregate detached fusion diagnostics once per train/validation epoch."""

    priority = 'LOW'

    @staticmethod
    def _get_fusion(runner: Runner):
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        if not hasattr(model, 'multimodal_fusion'):
            raise AttributeError('Model has no multimodal_fusion module.')
        fusion = model.multimodal_fusion
        required = ('start_statistics', 'stop_statistics', 'get_statistics')
        if any(not hasattr(fusion, name) for name in required):
            raise TypeError('Fusion module lacks the statistics interface.')
        return fusion

    def before_train_epoch(self, runner: Runner) -> None:
        self._get_fusion(runner).start_statistics()

    def after_train_epoch(self, runner: Runner) -> None:
        self._log(runner, 'train')

    def before_val(self, runner: Runner) -> None:
        self._get_fusion(runner).start_statistics()

    def after_val(self,
                  runner: Runner,
                  metrics: Optional[Dict[str, float]] = None) -> None:
        del metrics
        self._log(runner, 'val')

    def _log(self, runner: Runner, phase: str) -> None:
        fusion = self._get_fusion(runner)
        fusion.stop_statistics()
        statistics = fusion.get_statistics()
        epoch = runner.epoch + 1
        for key in sorted(statistics):
            accumulator = statistics[key]
            all_reduce(accumulator, op='sum')
            total, count = accumulator.tolist()
            if count <= 0 or get_rank() != 0:
                continue
            value = total / count
            runner.logger.info('Epoch(%s) [%d/%d]  fusion/%s: %.6f', phase,
                               epoch, runner.max_epochs, key, value)
            runner.visualizer.add_scalar(
                f'{phase}/fusion/{key}', value, step=epoch)
