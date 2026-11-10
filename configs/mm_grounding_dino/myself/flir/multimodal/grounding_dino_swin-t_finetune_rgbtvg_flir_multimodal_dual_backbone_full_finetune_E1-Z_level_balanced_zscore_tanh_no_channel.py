"""E1-Z for FLIR: per-level alignment z-score before tanh."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_E1-C_level_balanced_centered_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(
        # Replace E1-C centering with (A - mean) / std per feature level.
        center_alignment=False,
        alignment_normalization='zscore',
        alignment_normalization_eps=1e-6,
        spatial_gate_mode='dense',
        token_temperature=1.0,
        token_temperature_mode='fixed',
        gate_temperature=1.0,
        token_fpn_mode='level_balanced_shared',
        with_channel_gate=False))

work_dir = (
    'workdir/flir/multimodal/'
    'spatial_expectation_E1-Z_level_balanced_zscore_tanh_no_channel/'
    'groundingdino_rgbtvg_flir_E1-Z_train50')
