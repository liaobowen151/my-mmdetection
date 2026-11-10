"""E1-Z6T-T3: z-score six-state spatial fusion with tanh tau=3."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_E1-Z6T_level_balanced_zscore_six_state_tanh_tau1_no_channel.py'  # noqa: E501

model = dict(multimodal_fusion=dict(gate_temperature=3.0))
