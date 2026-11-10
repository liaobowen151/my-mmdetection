"""Swin Transformer with a switchable infrared-only LoRA branch."""

import math
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.swin import SwinTransformer
from mmdet.registry import MODELS


class IRLoRALinear(nn.Linear):
    """Linear layer whose LoRA residual is enabled only for IR forwards.

    There is deliberately no LoRA alpha or alpha/rank scaling. When enabled,
    the forward equation is ``y = xW^T + b + B(A(Dropout(x)))``.
    """

    def __init__(self,
                 in_features: int,
                 out_features: int,
                 lora_rank: int,
                 lora_dropout: float = 0.0,
                 bias: bool = True,
                 device=None,
                 dtype=None) -> None:
        super().__init__(in_features, out_features, bias=bias,
                         device=device, dtype=dtype)
        if lora_rank <= 0:
            raise ValueError(f'lora_rank must be positive, got {lora_rank}.')
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(
                f'lora_dropout must be in [0, 1), got {lora_dropout}.')

        self.lora_rank = int(lora_rank)
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = (nn.Dropout(self.lora_dropout_p)
                             if self.lora_dropout_p > 0.0 else nn.Identity())
        self.lora_A = nn.Parameter(torch.empty(
            self.lora_rank, in_features, device=device, dtype=dtype))
        self.lora_B = nn.Parameter(torch.empty(
            out_features, self.lora_rank, device=device, dtype=dtype))
        self.ir_lora_enabled = False
        self.reset_lora_parameters()

    @classmethod
    def from_linear(cls,
                    linear: nn.Linear,
                    lora_rank: int,
                    lora_dropout: float = 0.0) -> 'IRLoRALinear':
        module = cls(
            linear.in_features,
            linear.out_features,
            lora_rank=lora_rank,
            lora_dropout=lora_dropout,
            bias=linear.bias is not None,
            device=linear.weight.device,
            dtype=linear.weight.dtype)
        with torch.no_grad():
            module.weight.copy_(linear.weight)
            if linear.bias is not None:
                module.bias.copy_(linear.bias)
        module.weight.requires_grad_(linear.weight.requires_grad)
        if linear.bias is not None:
            module.bias.requires_grad_(linear.bias.requires_grad)
        module.train(linear.training)
        return module

    def reset_lora_parameters(self) -> None:
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        if not self.ir_lora_enabled:
            return result
        lora_input = self.lora_dropout(input)
        return result + F.linear(
            F.linear(lora_input, self.lora_A), self.lora_B)

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (f'{base}, lora_rank={self.lora_rank}, '
                f'ir_lora_enabled={self.ir_lora_enabled}, '
                f'lora_dropout={self.lora_dropout_p}')


@MODELS.register_module()
class IRLoRASwinTransformer(SwinTransformer):
    """Frozen shared Swin base with LoRA parameters for IR only."""

    def __init__(self,
                 *args,
                 lora_rank: int = 32,
                 lora_dropout: float = 0.1,
                 **kwargs) -> None:
        self.lora_rank = int(lora_rank)
        self.lora_dropout = float(lora_dropout)
        super().__init__(*args, **kwargs)
        self._inject_ir_lora()

    def _inject_ir_lora(self) -> None:
        for stage in self.stages:
            for block in stage.blocks:
                window_msa = block.attn.w_msa
                window_msa.qkv = IRLoRALinear.from_linear(
                    window_msa.qkv, self.lora_rank, self.lora_dropout)
                window_msa.proj = IRLoRALinear.from_linear(
                    window_msa.proj, self.lora_rank, self.lora_dropout)

    def set_ir_lora_enabled(self, enabled: bool) -> None:
        for module in self.modules():
            if isinstance(module, IRLoRALinear):
                module.ir_lora_enabled = bool(enabled)

    def set_lora_dropout_mode(self, training: bool) -> None:
        for module in self.modules():
            if isinstance(module, IRLoRALinear):
                module.lora_dropout.train(training)

    def freeze_base_and_enable_lora(self) -> None:
        self.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, IRLoRALinear):
                module.lora_A.requires_grad_(True)
                module.lora_B.requires_grad_(True)

    def uses_activation_checkpointing(self) -> bool:
        return any(block.with_cp for stage in self.stages
                   for block in stage.blocks)

    def trainable_lora_parameter_names(self) -> Sequence[str]:
        return tuple(name for name, parameter in self.named_parameters()
                     if parameter.requires_grad and
                     (name.endswith('.lora_A') or name.endswith('.lora_B')))
