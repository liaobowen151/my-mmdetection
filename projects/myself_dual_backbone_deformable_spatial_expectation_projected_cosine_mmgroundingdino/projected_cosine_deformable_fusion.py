"""Deformable VIS/IR enhancement followed by projected-cosine E1-Z."""

from mmdet.registry import MODELS
from projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino.deformable_spatial_expectation_fusion import (  # noqa: E501
    NoDirectTextDeformableSpatialExpectationFusionBlock,
)
from projects.myself_dual_backbone_spatial_channel_expectation_projected_cosine_mmgroundingdino.projected_cosine_fusion import (  # noqa: E501
    SpatialChannelExpectationProjectedCosineFusion,
)


@MODELS.register_module()
class NoDirectTextDeformableProjectedCosineSpatialExpectationFusionBlock(
        NoDirectTextDeformableSpatialExpectationFusionBlock):
    """Change only the E1-Z visual/token similarity implementation.

    The preceding bidirectional deformable VIS/IR attention is inherited
    unchanged.  The E1-Z block uses one visual projection shared by RGB and
    IR, one text projection, LayerNorm on visual features, L2 normalization
    on both sides, and a shared temperature-scaled cosine similarity.
    """

    def __init__(self,
                 embed_dims: int = 256,
                 num_heads: int = 8,
                 dropout: float = 0.0,
                 num_feature_levels: int = 4,
                 num_points: int = 4,
                 similarity_temperature: float = 1.0,
                 similarity_normalization_eps: float = 1e-6,
                 **expectation_kwargs) -> None:
        super().__init__(
            embed_dims=embed_dims,
            num_heads=num_heads,
            dropout=dropout,
            num_feature_levels=num_feature_levels,
            num_points=num_points,
            **expectation_kwargs)

        options = self.expectation_defaults()
        options.update(expectation_kwargs)
        self.spatial_expectation_fusion = (
            SpatialChannelExpectationProjectedCosineFusion(
                embed_dims=embed_dims,
                **options,
                similarity_temperature=similarity_temperature,
                similarity_normalization_eps=(
                    similarity_normalization_eps)))
