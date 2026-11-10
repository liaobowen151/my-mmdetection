"""Independent IR/VIS Hungarian assignments with a shared encoder."""

from mmdet.registry import MODELS
from projects.myself_dual_encoder_ir_guided_vis_mmgroundingdino import (
    DualEncoderIRGuidedVISGroundingDINOHead, )


@MODELS.register_module()
class SharedEncoderIndependentHungarianIRGuidedVISGroundingDINOHead(
        DualEncoderIRGuidedVISGroundingDINOHead):
    """Assign positives independently within the IR-selected Top-K set.

    Only the detector's encoder-sharing strategy differs from the dual-encoder
    experiment. The loss weights, matching costs, diagnostics and decoder
    losses remain identical so this version isolates the effect of sharing the
    encoder parameters.
    """

    pass
