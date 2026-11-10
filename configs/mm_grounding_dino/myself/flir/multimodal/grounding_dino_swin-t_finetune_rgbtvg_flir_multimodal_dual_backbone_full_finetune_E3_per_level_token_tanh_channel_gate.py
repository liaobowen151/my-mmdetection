"""E3: per-FPN-level token weights, tanh fusion, and channel gate."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_E2_level_balanced_tanh_channel_gate.py'  # noqa: E501

model = dict(multimodal_fusion=dict(token_fpn_mode='per_level'))
