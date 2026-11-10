"""Shared projected-cosine variant of spatial expectation fusion."""

# Keep the detector and statistics-hook registrations used by the parent
# E1-Z experiment while registering only a new, isolated fusion module here.
from projects.myself_dual_backbone_spatial_channel_expectation_mmgroundingdino import *  # noqa: F401,F403,E501
from projects.myself_multimodal_mmgroundingdino.datasets.paired_image_loading import (  # noqa: F401,E501
    LoadPairedImagesFromFile, )
from projects.myself_multimodal_mmgroundingdino.models.paired_data_preprocessor import (  # noqa: F401,E501
    PairedDetDataPreprocessor, )

from .projected_cosine_fusion import (
    SpatialChannelExpectationProjectedCosineFusion, )
from .hard_switch_hook import ProjectedCosineE1ZScheduleHook

__all__ = [
    'SpatialChannelExpectationProjectedCosineFusion',
    'ProjectedCosineE1ZScheduleHook',
    'LoadPairedImagesFromFile',
    'PairedDetDataPreprocessor',
]
