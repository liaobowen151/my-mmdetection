"""Freeze the pretrained feature/encoder path while adapting IR and decoder.

The frozen modules still take part in autograd.  In particular, the encoder is
not wrapped in ``torch.no_grad`` and its outputs are not detached, so decoder
losses can train the upstream IR LoRA and multimodal fusion modules.
"""

from mmdet.registry import MODELS
from projects.myself_ir_lora_addition_mmgroundingdino import (
    IRLoRAAdditionGroundingDINO,
)
from projects.myself_ir_lora_ir2vis_deformable_add_ir_mmgroundingdino import (
    IRLoRAIRToVISDeformableAddIRGroundingDINO,
)
from projects.myself_ir_lora_ir2vis_deformable_mmgroundingdino import (
    IRLoRAIRToVISDeformableGroundingDINO,
)


class _FrozenNeckEncoderMixin:
    """Shared freezing policy for the three IR-LoRA detector variants."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._freeze_pretrained_encoder_path()

    @staticmethod
    def _freeze_module(module) -> None:
        module.requires_grad_(False)
        module.eval()

    def _freeze_pretrained_encoder_path(self) -> None:
        if self.with_neck:
            self._freeze_module(self.neck)

        self._freeze_module(self.language_model)
        self._freeze_module(self.text_feat_map)
        self._freeze_module(self.encoder)

        self.level_embed.requires_grad_(False)
        self._freeze_module(self.memory_trans_fc)
        self._freeze_module(self.memory_trans_norm)

        # Grounding DINO appends one encoder proposal prediction branch after
        # the decoder-layer branches.  Keep only that final branch frozen;
        # decoder layers 0..num_layers-1 remain trainable.
        encoder_branch = self.decoder.num_layers
        if len(self.bbox_head.cls_branches) <= encoder_branch:
            raise RuntimeError('Missing encoder classification branch.')
        if len(self.bbox_head.reg_branches) <= encoder_branch:
            raise RuntimeError('Missing encoder regression branch.')
        self._freeze_module(self.bbox_head.cls_branches[encoder_branch])
        self._freeze_module(self.bbox_head.reg_branches[encoder_branch])

    def _keep_frozen_modules_in_eval(self) -> None:
        if self.with_neck:
            self.neck.eval()
        self.language_model.eval()
        self.text_feat_map.eval()
        self.encoder.eval()
        self.memory_trans_fc.eval()
        self.memory_trans_norm.eval()
        encoder_branch = self.decoder.num_layers
        self.bbox_head.cls_branches[encoder_branch].eval()
        self.bbox_head.reg_branches[encoder_branch].eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._keep_frozen_modules_in_eval()
        return self


@MODELS.register_module()
class FrozenNeckEncoderIRLoRAIRToVISDeformableAddIRGroundingDINO(
        _FrozenNeckEncoderMixin,
        IRLoRAIRToVISDeformableAddIRGroundingDINO):
    """IR-to-VIS deformable enhancement plus IR, with frozen neck/encoder."""


@MODELS.register_module()
class FrozenNeckEncoderIRLoRAIRToVISDeformableGroundingDINO(
        _FrozenNeckEncoderMixin,
        IRLoRAIRToVISDeformableGroundingDINO):
    """IR-to-VIS deformable enhancement, with frozen neck/encoder."""


@MODELS.register_module()
class FrozenNeckEncoderIRLoRAAdditionGroundingDINO(
        _FrozenNeckEncoderMixin,
        IRLoRAAdditionGroundingDINO):
    """Post-neck VIS/IR addition, with frozen neck/encoder."""
