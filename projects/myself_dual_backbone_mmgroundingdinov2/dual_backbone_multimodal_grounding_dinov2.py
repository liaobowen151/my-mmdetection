"""Multimodal Grounding DINO with independent VIS/IR backbones and necks."""

import copy
from typing import Dict

from torch import Tensor

from mmdet.registry import MODELS
from projects.myself_multimodal_mmgroundingdino.models.multimodal_grounding_dino import (  # noqa: E501
    MultimodalGroundingDINO,
)


@MODELS.register_module()
class DualBackboneMultimodalGroundingDINOv2(MultimodalGroundingDINO):
    """Use independent Swin backbones and ChannelMappers for VIS and IR.

    ``self.backbone`` and ``self.neck`` remain the VIS modules so official
    Grounding DINO checkpoint keys stay compatible. The IR modules start as
    deep copies. When a checkpoint lacks either IR key set, its corresponding
    VIS weights are mirrored to the IR module. This also allows a V1
    dual-backbone/shared-neck checkpoint to initialize ``self.ir_neck`` while
    preserving its already trained ``self.ir_backbone``.
    """

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

        self.ir_backbone = copy.deepcopy(self.backbone)
        self.ir_neck = copy.deepcopy(self.neck) if self.with_neck else None

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
        """Initialize missing IR backbone/neck keys from their VIS peers."""
        if module is not self:
            raise RuntimeError('Checkpoint pre-hook registered on wrong model.')

        prefix_pairs = [('backbone.', 'ir_backbone.')]
        if self.with_neck:
            prefix_pairs.append(('neck.', 'ir_neck.'))

        checkpoint_items = list(state_dict.items())
        for vis_suffix, ir_suffix in prefix_pairs:
            vis_prefix = prefix + vis_suffix
            ir_prefix = prefix + ir_suffix
            if any(key.startswith(ir_prefix) for key in state_dict):
                continue

            mirrored = {}
            for key, value in checkpoint_items:
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
        """Run VIS and IR through independent backbones and necks."""
        if batch_inputs.ndim != 4 or batch_inputs.size(1) != 6:
            raise ValueError(
                'DualBackboneMultimodalGroundingDINOv2 expects batch_inputs '
                f'with shape [B, 6, H, W], got {tuple(batch_inputs.shape)}.')

        vis_inputs = batch_inputs[:, :3]
        ir_inputs = batch_inputs[:, 3:]

        vis_feats = self.backbone(vis_inputs)
        ir_feats = self.ir_backbone(ir_inputs)

        if self.with_neck:
            vis_feats = self.neck(vis_feats)
            ir_feats = self.ir_neck(ir_feats)

        if len(vis_feats) != len(ir_feats):
            raise RuntimeError(
                'VIS and IR backbones/necks returned different feature '
                f'counts: {len(vis_feats)} versus {len(ir_feats)}.')
        for level, (vis_feat, ir_feat) in enumerate(
                zip(vis_feats, ir_feats)):
            if vis_feat.shape != ir_feat.shape:
                raise RuntimeError(
                    f'VIS/IR feature shape mismatch at level {level}: '
                    f'{tuple(vis_feat.shape)} versus {tuple(ir_feat.shape)}.')

        return dict(
            vis_feats=tuple(vis_feats),
            ir_feats=tuple(ir_feats))
