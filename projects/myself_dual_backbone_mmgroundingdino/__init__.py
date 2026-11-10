"""Independent dual-backbone extension for multimodal Grounding DINO."""

# Register the paired loader, preprocessor and multimodal fusion components.
from projects.myself_multimodal_mmgroundingdino import *  # noqa: F401,F403

from .dual_backbone_multimodal_grounding_dino import (
    DualBackboneMultimodalGroundingDINO)

__all__ = ['DualBackboneMultimodalGroundingDINO']
