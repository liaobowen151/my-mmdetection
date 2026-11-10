"""E1-Z: replace E1-C centering with per-level alignment z-score."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_E1-C_level_balanced_centered_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(
        # The only model change from E1-C is (A - mean) -> (A - mean) / std.
        center_alignment=False,
        alignment_normalization='zscore',
        alignment_normalization_eps=1e-6,
        spatial_gate_mode='dense',
        token_temperature=1.0,
        token_temperature_mode='fixed',
        gate_temperature=1.0,
        token_fpn_mode='level_balanced_shared',
        with_channel_gate=False))
