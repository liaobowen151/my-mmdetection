"""E1: M3FD level-balanced shared-token tanh fusion, no channel gate."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_base_spatial_expectation_fusion.py'  # noqa: E501

custom_imports = dict(
    imports=[
        'projects.'
        'myself_dual_backbone_spatial_channel_expectation_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneSpatialChannelExpectationGroundingDINO',
    multimodal_fusion=dict(
        _delete_=True,
        type='SpatialChannelExpectationFusion',
        embed_dims=256,
        token_temperature=1.0,
        gate_temperature=1.0,
        token_fpn_mode='level_balanced_shared',
        with_channel_gate=False,
        channel_reduction=4))

custom_hooks = [dict(type='SpatialChannelExpectationStatisticsHook')]
