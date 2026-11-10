"""Zero-shot Grounding DINO with shared-backbone post-neck addition."""

from typing import Tuple

import torch
from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


@MODELS.register_module()
class SharedBackboneNeckAddZeroShotGroundingDINO(GroundingDINO):
    """Fuse paired VIS/IR features by addition after one shared neck.

    The module deliberately introduces no modality-specific or fusion
    parameters. VIS and IR are stacked along the batch dimension, processed
    by the same backbone and neck, split back into their respective branches,
    and added level by level. This preserves the parameter names of official
    Grounding DINO so its pretrained checkpoint can initialize the complete
    learnable model.
    """

    def __init__(self,
                 *args,
                 freeze_visual_backbone: bool = True,
                 freeze_language_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model
        self._freeze_pretrained_modules()

    def _freeze_pretrained_modules(self) -> None:
        if self.freeze_visual_backbone:
            self.backbone.requires_grad_(False)
            self.backbone.eval()
        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            self.backbone.eval()
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """Extract and add post-neck VIS/IR features."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'SharedBackboneNeckAddZeroShotGroundingDINO expects inputs '
                f'with shape [B, 6, H, W], got {tuple(batch_inputs.shape)}.')

        batch_size = batch_inputs.size(0)
        paired_inputs = torch.cat(
            [batch_inputs[:, :3], batch_inputs[:, 3:]], dim=0)
        paired_feats = self.backbone(paired_inputs)
        if self.with_neck:
            paired_feats = self.neck(paired_feats)

        fused_feats = []
        for level, paired_feat in enumerate(paired_feats):
            expected_size = 2 * batch_size
            if paired_feat.size(0) != expected_size:
                raise RuntimeError(
                    f'Feature level {level} has batch size '
                    f'{paired_feat.size(0)}, expected {expected_size}.')
            vis_feat, ir_feat = paired_feat.split(batch_size, dim=0)
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'Feature shape mismatch at level {level}: VIS '
                    f'{tuple(vis_feat.shape)} versus IR '
                    f'{tuple(ir_feat.shape)}.')
            fused_feats.append(vis_feat + ir_feat)

        return tuple(fused_feats)
