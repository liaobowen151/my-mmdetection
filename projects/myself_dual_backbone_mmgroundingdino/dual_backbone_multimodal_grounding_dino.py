"""Multimodal Grounding DINO with independent VIS and IR backbones."""

import copy
from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_multimodal_mmgroundingdino.models.multimodal_grounding_dino import (  # noqa: E501
    MultimodalGroundingDINO,
)


@MODELS.register_module()
class DualBackboneMultimodalGroundingDINO(MultimodalGroundingDINO):
    """Use separately trainable Swin backbones for VIS and IR.

    ``self.backbone`` remains the VIS backbone so official Grounding DINO
    checkpoint keys stay compatible. ``self.ir_backbone`` starts as a deep
    copy. During initial official-checkpoint loading, missing IR keys are
    mirrored from their VIS counterparts. A checkpoint produced by this model
    already contains both key sets and is therefore restored without mirroring.

    The ChannelMapper neck remains shared between the two modalities.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ir_backbone = copy.deepcopy(self.backbone)

        if not self.freeze_visual_backbone:
            self.backbone.requires_grad_(True)
            self.ir_backbone.requires_grad_(True)
        else:
            self.ir_backbone.requires_grad_(False)
            self.ir_backbone.eval()

        # This project currently runs on a PyTorch version that exposes the
        # module-aware variant through the underscored compatibility API.
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
        """Initialize missing IR keys from VIS without breaking resume."""
        if module is not self:
            raise RuntimeError('Checkpoint pre-hook registered on wrong model.')

        vis_prefix = prefix + 'backbone.'
        ir_prefix = prefix + 'ir_backbone.'
        if any(key.startswith(ir_prefix) for key in state_dict):
            return

        mirrored = {}
        for key, value in state_dict.items():
            if key.startswith(vis_prefix):
                suffix = key[len(vis_prefix):]
                mirrored[ir_prefix + suffix] = value.clone()
        state_dict.update(mirrored)

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_visual_backbone:
            self.ir_backbone.eval()
        return self

    def extract_feat(self, batch_inputs: Tensor) -> Dict:
        """Run VIS and IR through their independent Swin backbones."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualBackboneMultimodalGroundingDINO expects batch_inputs '
                f'with shape [B, 6, H, W], got {tuple(batch_inputs.shape)}.')

        vis_inputs = batch_inputs[:, :3]
        ir_inputs = batch_inputs[:, 3:]

        vis_feats = self.backbone(vis_inputs)
        ir_feats = self.ir_backbone(ir_inputs)

        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError(
                'VIS and IR backbones returned different feature counts: '
                f'{len(vis_feats)} versus {len(ir_feats)}.')
        for level, (vis_feat, ir_feat) in enumerate(
                zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'VIS/IR feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} versus {tuple(ir_feat.shape)}.')

        return dict(
            vis_feats=tuple(vis_feats),
            ir_feats=tuple(ir_feats))
