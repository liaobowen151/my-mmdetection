"""Grounding DINO using two forwards through one dual-LoRA Swin backbone."""

from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_multimodal_mmgroundingdino.models.multimodal_grounding_dino import (  # noqa: E501
    MultimodalGroundingDINO,
)

from .dual_lora_swin import DualAdapterLoRASwinTransformer


@MODELS.register_module()
class DualLoRAMultimodalGroundingDINO(MultimodalGroundingDINO):
    """Multimodal Grounding DINO with selectable RGB/IR LoRA adapters."""

    def _freeze_pretrained_backbones(self) -> None:
        if self.freeze_visual_backbone:
            if not isinstance(self.backbone,
                              DualAdapterLoRASwinTransformer):
                raise TypeError(
                    'DualLoRAMultimodalGroundingDINO requires backbone type '
                    'DualAdapterLoRASwinTransformer when '
                    'freeze_visual_backbone=True.')
            if self.backbone.uses_activation_checkpointing():
                raise ValueError(
                    'Dynamic RGB/IR adapter switching is incompatible with '
                    'Swin activation checkpoint recomputation. Set '
                    'model.backbone.with_cp=False.')
            self.backbone.freeze_base_and_enable_lora()
            self.backbone.eval()

        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            # Keep frozen Swin stochastic layers such as DropPath disabled,
            # then independently restore LoRA dropout during training.
            self.backbone.eval()
            self.backbone.set_lora_dropout_mode(mode)
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        """Run RGB and IR sequentially with their respective adapter."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualLoRAMultimodalGroundingDINO expects batch_inputs with '
                f'shape [B, 6, H, W], but got {tuple(batch_inputs.shape)}.')

        vis_inputs = batch_inputs[:, :3]
        ir_inputs = batch_inputs[:, 3:]

        self.backbone.set_adapter('rgb')
        vis_feats = self.backbone(vis_inputs)
        if self.with_neck:
            vis_feats = self.neck(vis_feats)

        self.backbone.set_adapter('ir')
        ir_feats = self.backbone(ir_inputs)
        if self.with_neck:
            ir_feats = self.neck(ir_feats)

        # Leave a stable default state for inspection and RGB-only utilities.
        self.backbone.set_adapter('rgb')
        return dict(vis_feats=tuple(vis_feats), ir_feats=tuple(ir_feats))
