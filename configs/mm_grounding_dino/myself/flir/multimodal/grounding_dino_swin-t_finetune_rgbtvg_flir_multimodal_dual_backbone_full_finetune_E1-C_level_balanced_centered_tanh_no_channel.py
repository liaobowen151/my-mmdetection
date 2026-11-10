"""E1-C: center per-level alignment before tanh; fixed temperatures."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_E1_level_balanced_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(
        center_alignment=True,
        token_temperature=1.0,
        token_temperature_mode='fixed',
        gate_temperature=1.0))
