"""Epoch schedule shared by IR injection and interest loss."""

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper

from mmdet.registry import HOOKS


@HOOKS.register_module()
class TextFusionInterestScheduleHook(Hook):
    priority = 'VERY_HIGH'

    def __init__(self, lambda_interest_max: float) -> None:
        if lambda_interest_max <= 0:
            raise ValueError('lambda_interest_max must be positive.')
        self.lambda_interest_max = float(lambda_interest_max)

    @staticmethod
    def _alpha(epoch: int) -> float:
        return 0.1 + 0.18 * (epoch - 1) if epoch <= 5 else 1.0

    def before_train_epoch(self, runner) -> None:
        model = runner.model.module if is_model_wrapper(
            runner.model) else runner.model
        epoch = runner.epoch + 1
        alpha = self._alpha(epoch)
        interest_weight = alpha * self.lambda_interest_max
        model.multimodal_fusion.set_schedule(alpha, interest_weight)
        runner.logger.info(
            'TextFusion schedule epoch=%d: alpha=%.2f, '
            'lambda_interest=%.3f', epoch, alpha, interest_weight)
