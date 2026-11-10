"""E1-Z6T-T1: z-score six-state spatial fusion with tanh tau=1."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_E1_level_balanced_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(
        center_alignment=False,
        alignment_normalization='zscore',
        alignment_normalization_eps=1e-6,
        spatial_gate_mode='six_state',
        high_response_threshold=1.5,
        low_response_threshold=-0.5,
        token_temperature=1.0,
        token_temperature_mode='fixed',
        gate_temperature=1.0,
        token_fpn_mode='level_balanced_shared',
        with_channel_gate=False))
