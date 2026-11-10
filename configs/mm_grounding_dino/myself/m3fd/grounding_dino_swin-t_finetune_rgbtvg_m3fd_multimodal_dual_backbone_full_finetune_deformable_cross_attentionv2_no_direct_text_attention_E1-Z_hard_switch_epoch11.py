"""M3FD: activate no-channel E1-Z after epoch 10."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_deformable_cross_attentionv2_no_text_attention.py'

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_deformable_spatial_expectation_hard_switch_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneNoDirectTextDeformableSpatialExpectationHardSwitchGroundingDINO',  # noqa: E501
    multimodal_fusion=dict(
        _delete_=True,
        type='NoDirectTextDeformableSpatialExpectationHardSwitchFusionBlock',
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

custom_hooks = [
    dict(type='HardSwitchFusionScheduleHook', start_epoch=11),
    dict(type='SpatialChannelExpectationStatisticsHook'),
]

work_dir = ('workdir/m3fd/multimodal/'
            'dual_backbone_deformable_cross_attentionv2_no_direct_text_'
            'attention_E1-Z_hard_switch_epoch11/'
            'groundingdino_rgbtvg_m3fd_no_direct_text_deformable_E1-Z_'
            'hard_switch_epoch11_train50')
