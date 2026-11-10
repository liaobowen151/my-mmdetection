"""No-direct-text deformable cross-attention followed by E1-Z fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_deformable_cross_attentionv2_no_text_attention.py'

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_deformable_spatial_expectation_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO',
    multimodal_fusion=dict(
        type='NoDirectTextDeformableSpatialExpectationFusionBlock',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        token_fpn_mode='level_balanced_shared',
        alignment_normalization='zscore',
        alignment_normalization_eps=1e-6,
        center_alignment=False,
        spatial_gate_mode='dense',
        gate_temperature=1.0,
        token_temperature=1.0,
        token_temperature_mode='fixed',
        with_channel_gate=False,
        channel_reduction=4))

custom_hooks = [dict(type='SpatialChannelExpectationStatisticsHook')]

work_dir = ('workdir/flir/multimodal/'
            'dual_backbone_deformable_cross_attentionv2_no_direct_text_'
            'attention_E1-Z_level_balanced_zscore_tanh_no_channel/'
            'groundingdino_rgbtvg_flir_no_direct_text_deformable_E1-Z_'
            'train50')
