"""Direct-addition baseline for dual-backbone Grounding DINO."""

# Reuse only the paired-input data components. The detector implementation
# itself is independent from the existing multimodal detector classes.
from projects.myself_multimodal_mmgroundingdino.datasets.paired_image_loading import (  # noqa: F401,E501
    LoadPairedImagesFromFile,
)
from projects.myself_multimodal_mmgroundingdino.models.paired_data_preprocessor import (  # noqa: F401,E501
    PairedDetDataPreprocessor,
)

from .dual_backbone_base_grounding_dino import DualBackboneBaseGroundingDINO

__all__ = [
    'DualBackboneBaseGroundingDINO', 'LoadPairedImagesFromFile',
    'PairedDetDataPreprocessor'
]
