"""E4: per-FPN-level token weights and tanh fusion, no channel gate."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_E1_level_balanced_tanh_no_channel.py'  # noqa: E501

model = dict(multimodal_fusion=dict(token_fpn_mode='per_level'))
