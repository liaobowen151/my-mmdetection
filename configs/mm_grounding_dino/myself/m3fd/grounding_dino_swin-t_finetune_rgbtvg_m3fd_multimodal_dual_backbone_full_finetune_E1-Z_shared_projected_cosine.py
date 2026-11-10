"""M3FD: base fusion for 10 epochs, then projected-cosine E1-Z."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_base.py'  # noqa: E501

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_spatial_channel_expectation_'
        'projected_cosine_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneSpatialChannelExpectationGroundingDINO',
    multimodal_fusion=dict(
        type='SpatialChannelExpectationProjectedCosineFusion',
        embed_dims=256,
        token_temperature=1.0,
        gate_temperature=1.0,
        token_fpn_mode='level_balanced_shared',
        with_channel_gate=False,
        channel_reduction=4,
        center_alignment=False,
        alignment_normalization='zscore',
        alignment_normalization_eps=1e-6,
        spatial_gate_mode='dense',
        token_temperature_mode='fixed',
        similarity_temperature=1.0,
        similarity_normalization_eps=1e-6,
        enable_hard_switch=True))

custom_hooks = [
    dict(type='ProjectedCosineE1ZScheduleHook', start_epoch=11),
    dict(type='SpatialChannelExpectationStatisticsHook'),
]

work_dir = ('workdir/m3fd/multimodal/'
            'spatial_expectation_E1-Z_shared_projected_cosine/'
            'groundingdino_rgbtvg_m3fd_E1-Z_shared_projected_cosine_train50')
