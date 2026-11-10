"""Dual-backbone Grounding DINO baseline with direct feature addition."""

import copy
from typing import Tuple

from torch import Tensor

from mmdet.models.detectors.grounding_dino import GroundingDINO
from mmdet.registry import MODELS


@MODELS.register_module()
class DualBackboneBaseGroundingDINO(GroundingDINO):
    """Fuse independent VIS/IR backbone features by element-wise addition.

    VIS and IR use independent Swin backbones and share one ChannelMapper. The
    corresponding post-neck feature levels are added directly before entering
    the unmodified Grounding DINO Transformer. No custom text-guided visual
    enhancement or cross-modal attention module is used.
    """

    def __init__(self,
                 *args,
                 freeze_visual_backbone: bool = False,
                 freeze_language_model: bool = True,
                 **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ir_backbone = copy.deepcopy(self.backbone)
        self.freeze_visual_backbone = freeze_visual_backbone
        self.freeze_language_model = freeze_language_model

        if self.freeze_visual_backbone:
            self.backbone.requires_grad_(False)
            self.ir_backbone.requires_grad_(False)
            self.backbone.eval()
            self.ir_backbone.eval()
        else:
            self.backbone.requires_grad_(True)
            self.ir_backbone.requires_grad_(True)

        if self.freeze_language_model:
            self.language_model.requires_grad_(False)
            self.language_model.eval()
        else:
            self.language_model.requires_grad_(True)
        # Match the source experiment: BERT is frozen, while the projection
        # from language-model dimensions to the detector dimensions is trained.
        self.text_feat_map.requires_grad_(True)

        self._register_load_state_dict_pre_hook(
            self._mirror_vis_checkpoint_to_ir, with_module=True)

    def _mirror_vis_checkpoint_to_ir(
        self,
        module,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Initialize a missing IR backbone from the VIS checkpoint weights."""
        if module is not self:
            raise RuntimeError('Checkpoint pre-hook registered on wrong model.')

        vis_prefix = prefix + 'backbone.'
        ir_prefix = prefix + 'ir_backbone.'
        if any(key.startswith(ir_prefix) for key in state_dict):
            return

        mirrored = {}
        for key, value in list(state_dict.items()):
            if key.startswith(vis_prefix):
                suffix = key[len(vis_prefix):]
                mirrored[ir_prefix + suffix] = value.clone()
        state_dict.update(mirrored)

    def train(self, mode: bool = True):
        """Keep frozen modules in evaluation mode during training."""
        super().train(mode)
        if self.freeze_visual_backbone:
            self.backbone.eval()
            self.ir_backbone.eval()
        if self.freeze_language_model:
            self.language_model.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Tuple[Tensor, ...]:
        """Extract, map and directly add paired VIS/IR feature levels."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualBackboneBaseGroundingDINO expects batch_inputs with '
                f'shape [B, 6, H, W], got {tuple(batch_inputs.shape)}.')

        vis_inputs = batch_inputs[:, :3]
        ir_inputs = batch_inputs[:, 3:]

        vis_feats = self.backbone(vis_inputs)
        ir_feats = self.ir_backbone(ir_inputs)

        if self.with_neck:
            # Intentionally call the same ChannelMapper for both modalities.
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError(
                'VIS and IR branches returned different feature counts: '
                f'{len(vis_feats)} versus {len(ir_feats)}.')

        fused_feats = []
        for level, (vis_feat, ir_feat) in enumerate(zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'VIS/IR feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} versus {tuple(ir_feat.shape)}.')
            fused_feats.append(vis_feat + ir_feat)

        return tuple(fused_feats)
