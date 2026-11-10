"""E3: M3FD per-level token tanh fusion with channel gate."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_E2_level_balanced_tanh_channel_gate.py'  # noqa: E501

model = dict(multimodal_fusion=dict(token_fpn_mode='per_level'))
