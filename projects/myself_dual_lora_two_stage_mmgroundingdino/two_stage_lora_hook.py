"""Freeze dual attention LoRA for a warm-up stage, then enable it."""

from typing import List, Tuple

from mmengine.hooks import Hook
from mmengine.model import is_model_wrapper
from mmengine.registry import HOOKS
from torch.nn import Parameter

from projects.myself_dual_lora_mmgroundingdino.dual_lora_swin import (
    DualAdapterLoRALinear,
)


@HOOKS.register_module()
class TwoStageLoRAFreezeHook(Hook):
    """Control both QKV/proj RGB/IR LoRA adapters by epoch.

    The LoRA parameters deliberately remain trainable while the model, DDP
    wrapper and optimizer are constructed. This guarantees that DDP registers
    their reduction hooks and that the optimizer owns the parameters. The hook
    then freezes them before the first forward of epochs 1--8 and enables them
    before the first forward of epoch 9.

    Args:
        unfreeze_epoch: Zero-based epoch index at which LoRA becomes
            trainable. The default value 8 means user-facing epoch 9.
        expected_lora_layers: Expected number of LoRA-wrapped attention
            linear layers. Swin-T has 12 blocks and wraps both QKV and proj,
            giving 24 layers. Set to ``None`` to skip this check.
    """

    priority = 'VERY_HIGH'

    def __init__(self,
                 unfreeze_epoch: int = 8,
                 expected_lora_layers: int = 24) -> None:
        if unfreeze_epoch < 0:
            raise ValueError('unfreeze_epoch must be non-negative.')
        if expected_lora_layers is not None and expected_lora_layers <= 0:
            raise ValueError('expected_lora_layers must be positive or None.')
        self.unfreeze_epoch = unfreeze_epoch
        self.expected_lora_layers = expected_lora_layers
        self._last_trainable = None

    @staticmethod
    def _unwrap_model(runner):
        model = runner.model
        return model.module if is_model_wrapper(model) else model

    def _lora_parameters(self, runner) -> List[Tuple[str, Parameter]]:
        model = self._unwrap_model(runner)
        if not hasattr(model, 'backbone'):
            raise TypeError('TwoStageLoRAFreezeHook requires model.backbone.')

        lora_layers = [
            module for module in model.backbone.modules()
            if isinstance(module, DualAdapterLoRALinear)
        ]
        if self.expected_lora_layers is not None and len(lora_layers) != \
                self.expected_lora_layers:
            raise RuntimeError(
                'Unexpected number of dual-LoRA layers: expected '
                f'{self.expected_lora_layers}, got {len(lora_layers)}.')

        parameters = []
        for name, parameter in model.backbone.named_parameters():
            if '.lora_A.' in name or '.lora_B.' in name:
                parameters.append((name, parameter))
        if not parameters:
            raise RuntimeError('No RGB/IR LoRA A/B parameters were found.')
        return parameters

    def before_train(self, runner) -> None:
        """Verify that every LoRA tensor is already owned by the optimizer."""
        lora_parameters = self._lora_parameters(runner)
        optimizer = runner.optim_wrapper.optimizer
        optimized_ids = {
            id(parameter)
            for group in optimizer.param_groups
            for parameter in group['params']
        }
        missing = [
            name for name, parameter in lora_parameters
            if id(parameter) not in optimized_ids
        ]
        if missing:
            raise RuntimeError(
                'LoRA parameters must enter the optimizer before stage-1 '
                f'freezing, but these parameters are missing: {missing}.')

    def before_train_epoch(self, runner) -> None:
        """Freeze in epochs 1--8 and enable from epoch 9 onward."""
        trainable = runner.epoch >= self.unfreeze_epoch
        lora_parameters = self._lora_parameters(runner)
        for _, parameter in lora_parameters:
            parameter.requires_grad_(trainable)

        if self._last_trainable != trainable:
            user_epoch = runner.epoch + 1
            state = 'trainable' if trainable else 'frozen'
            runner.logger.info(
                'Two-stage dual LoRA: RGB/IR QKV/proj LoRA is %s from '
                'epoch %d '
                '(%d parameter tensors).', state, user_epoch,
                len(lora_parameters))
            self._last_trainable = trainable
