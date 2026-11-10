"""E2: M3FD level-balanced shared-token tanh fusion with channel gate."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_E1_level_balanced_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(with_channel_gate=True, channel_reduction=4))
