"""Grounding DINO with shared frozen Swin, IR LoRA and direct addition."""

from typing import Tuple

from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS

from .ir_lora_swin import IRLoRASwinTransformer


@MODELS.register_module()
class IRLoRAAdditionGroundingDINO(GroundingDINO):
    """Add post-neck RGB-base and IR-LoRA features before the encoder."""

    def __init__(self,
                 *args,
                 freeze_visual_backbone: bool = True,
                 freeze_language_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model

        if not isinstance(self.backbone, IRLoRASwinTransformer):
            raise TypeError('IRLoRAAdditionGroundingDINO requires an '
                            'IRLoRASwinTransformer backbone.')
        if not self.freeze_visual_backbone:
            raise ValueError('The shared base backbone must be frozen.')
        if self.backbone.uses_activation_checkpointing():
            raise ValueError('Adapter switching requires with_cp=False.')

        self.backbone.freeze_base_and_enable_lora()
        self.backbone.eval()
        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
        self.text_feat_map.requires_grad_(True)

    def train(self, mode: bool = True):
        super().train(mode)
        self.backbone.eval()
        self.backbone.set_lora_dropout_mode(mode)
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'IRLoRAAdditionGroundingDINO expects [B, 6, H, W], got '
                f'{tuple(batch_inputs.shape)}.')

        self.backbone.set_ir_lora_enabled(False)
        vis_feats = self.backbone(batch_inputs[:, :3])

        self.backbone.set_ir_lora_enabled(True)
        ir_feats = self.backbone(batch_inputs[:, 3:])

        # Keep a safe default: an unrelated backbone call remains RGB/base-only.
        self.backbone.set_ir_lora_enabled(False)

        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError('RGB and IR branches returned different levels.')
        fused_feats = []
        for level, (vis_feat, ir_feat) in enumerate(zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'Feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} vs {tuple(ir_feat.shape)}.')
            fused_feats.append(vis_feat + ir_feat)
        return tuple(fused_feats)
