"""Swin Transformer with independent RGB and infrared LoRA adapters.

The original Swin weights remain under their existing state-dict keys. This
lets checkpoints produced for ``mmdet.models.SwinTransformer`` initialize the
base weights directly; only the newly added LoRA parameters are missing.
"""

import math
from typing import Dict, Mapping, Sequence, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmdet.models.backbones.swin import SwinTransformer
from mmdet.registry import MODELS


class DualAdapterLoRALinear(nn.Linear):
    r"""A Linear layer with two selectable LoRA adapters.

    The forward equation for the active adapter is

    .. math:: y = xW^T + b + (\alpha / r) B(A(\mathrm{Dropout}(x))).

    ``weight`` and ``bias`` deliberately stay on this module (instead of a
    nested ``base_layer``), preserving the checkpoint names of ``nn.Linear``.
    """

    ADAPTER_NAMES = ('rgb', 'ir')

    def __init__(
        self,
        in_features: int,
        out_features: int,
        ranks: Mapping[str, int],
        alphas: Union[float, Mapping[str, float]],
        lora_dropout: float = 0.0,
        bias: bool = True,
        device=None,
        dtype=None,
    ) -> None:
        super().__init__(
            in_features,
            out_features,
            bias=bias,
            device=device,
            dtype=dtype)

        self.ranks = self._validate_ranks(ranks)
        self.alphas = self._normalize_alphas(alphas)
        if not 0.0 <= lora_dropout < 1.0:
            raise ValueError(
                'lora_dropout must be in [0, 1), but got '
                f'{lora_dropout}.')
        self.lora_dropout_p = float(lora_dropout)
        self.lora_dropout = nn.Dropout(self.lora_dropout_p) \
            if self.lora_dropout_p > 0.0 else nn.Identity()
        self.scalings = {
            name: self.alphas[name] / self.ranks[name]
            for name in self.ADAPTER_NAMES
        }
        self.lora_A = nn.ParameterDict()
        self.lora_B = nn.ParameterDict()
        for name in self.ADAPTER_NAMES:
            rank = self.ranks[name]
            self.lora_A[name] = nn.Parameter(
                torch.empty(rank, in_features, device=device, dtype=dtype))
            self.lora_B[name] = nn.Parameter(
                torch.empty(out_features, rank, device=device, dtype=dtype))

        self.active_adapter = 'rgb'
        self.reset_lora_parameters()

    @classmethod
    def from_linear(
        cls,
        linear: nn.Linear,
        ranks: Mapping[str, int],
        alphas: Union[float, Mapping[str, float]],
        lora_dropout: float = 0.0,
    ) -> 'DualAdapterLoRALinear':
        """Create a LoRA layer without changing the base linear output."""
        module = cls(
            linear.in_features,
            linear.out_features,
            ranks=ranks,
            alphas=alphas,
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

    @classmethod
    def _validate_ranks(cls, ranks: Mapping[str, int]) -> Dict[str, int]:
        if set(ranks) != set(cls.ADAPTER_NAMES):
            raise ValueError(
                'ranks must contain exactly the adapters '
                f'{cls.ADAPTER_NAMES}, but got {tuple(ranks)}.')
        normalized = {name: int(ranks[name]) for name in cls.ADAPTER_NAMES}
        if any(rank <= 0 for rank in normalized.values()):
            raise ValueError(f'All LoRA ranks must be positive: {normalized}.')
        return normalized

    @classmethod
    def _normalize_alphas(
        cls,
        alphas: Union[float, Mapping[str, float]],
    ) -> Dict[str, float]:
        if isinstance(alphas, Mapping):
            if set(alphas) != set(cls.ADAPTER_NAMES):
                raise ValueError(
                    'alphas must contain exactly the adapters '
                    f'{cls.ADAPTER_NAMES}, but got {tuple(alphas)}.')
            return {name: float(alphas[name]) for name in cls.ADAPTER_NAMES}
        return {name: float(alphas) for name in cls.ADAPTER_NAMES}

    def reset_lora_parameters(self) -> None:
        """Use the standard LoRA initialization: random A and zero B."""
        for name in self.ADAPTER_NAMES:
            nn.init.kaiming_uniform_(self.lora_A[name], a=math.sqrt(5))
            nn.init.zeros_(self.lora_B[name])

    def set_adapter(self, adapter_name: str) -> None:
        if adapter_name not in self.ADAPTER_NAMES:
            raise ValueError(
                f'Unknown adapter {adapter_name!r}; expected one of '
                f'{self.ADAPTER_NAMES}.')
        self.active_adapter = adapter_name

    def set_lora_trainable(self, trainable: bool) -> None:
        """Toggle both adapters without changing the frozen base weight."""
        for name in self.ADAPTER_NAMES:
            self.lora_A[name].requires_grad_(trainable)
            self.lora_B[name].requires_grad_(trainable)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        result = F.linear(input, self.weight, self.bias)
        name = self.active_adapter
        lora_input = self.lora_dropout(input)
        delta = F.linear(F.linear(lora_input, self.lora_A[name]),
                         self.lora_B[name])
        return result + delta * self.scalings[name]

    def extra_repr(self) -> str:
        base = super().extra_repr()
        return (f'{base}, ranks={self.ranks}, alphas={self.alphas}, '
                f'active_adapter={self.active_adapter!r}, '
                f'lora_dropout={self.lora_dropout_p}')


@MODELS.register_module()
class DualAdapterLoRASwinTransformer(SwinTransformer):
    """Swin with separate RGB/IR LoRA in every attention QKV and proj."""

    def __init__(
        self,
        *args,
        lora_ranks: Mapping[str, int] = dict(rgb=16, ir=48),
        lora_alpha: Union[float, Mapping[str, float]] = dict(
            rgb=16., ir=16.),
        lora_dropout: float = 0.0,
        **kwargs,
    ) -> None:
        self.lora_ranks = dict(lora_ranks)
        self.lora_alpha = lora_alpha
        self.lora_dropout = float(lora_dropout)
        self.active_adapter = 'rgb'
        super().__init__(*args, **kwargs)
        self._inject_attention_lora()

    def _inject_attention_lora(self) -> None:
        for stage in self.stages:
            for block in stage.blocks:
                window_msa = block.attn.w_msa
                window_msa.qkv = DualAdapterLoRALinear.from_linear(
                    window_msa.qkv,
                    ranks=self.lora_ranks,
                    alphas=self.lora_alpha,
                    lora_dropout=self.lora_dropout)
                window_msa.proj = DualAdapterLoRALinear.from_linear(
                    window_msa.proj,
                    ranks=self.lora_ranks,
                    alphas=self.lora_alpha,
                    lora_dropout=self.lora_dropout)

    def set_adapter(self, adapter_name: str) -> None:
        """Select one adapter for the next complete backbone forward."""
        for module in self.modules():
            if isinstance(module, DualAdapterLoRALinear):
                module.set_adapter(adapter_name)
        self.active_adapter = adapter_name

    def set_lora_dropout_mode(self, training: bool) -> None:
        """Set only LoRA dropout mode while the frozen Swin stays in eval."""
        for module in self.modules():
            if isinstance(module, DualAdapterLoRALinear):
                module.lora_dropout.train(training)

    def freeze_base_and_enable_lora(self) -> None:
        """Freeze original Swin and enable all attention LoRA adapters."""
        self.requires_grad_(False)
        for module in self.modules():
            if isinstance(module, DualAdapterLoRALinear):
                module.set_lora_trainable(True)

    def uses_activation_checkpointing(self) -> bool:
        """Whether any Swin block may be recomputed during backward."""
        return any(block.with_cp for stage in self.stages
                   for block in stage.blocks)

    def trainable_lora_parameter_names(self) -> Sequence[str]:
        return tuple(
            name for name, parameter in self.named_parameters()
            if parameter.requires_grad and
            ('.lora_A.' in name or '.lora_B.' in name))
