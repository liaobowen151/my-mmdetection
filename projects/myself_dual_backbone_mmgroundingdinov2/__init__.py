"""Independent dual-backbone, dual-neck multimodal Grounding DINO."""

# Register the paired loader, preprocessor and multimodal fusion components.
from projects.myself_multimodal_mmgroundingdino import *  # noqa: F401,F403

from .dual_backbone_multimodal_grounding_dinov2 import (
    DualBackboneMultimodalGroundingDINOv2)

__all__ = ['DualBackboneMultimodalGroundingDINOv2']
