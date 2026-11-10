"""E1-CLT: original E1 with valid-text-length token temperature."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_E1_level_balanced_tanh_no_channel.py'  # noqa: E501

model = dict(
    multimodal_fusion=dict(
        # Keep the original E1 alignment: CLT does not perform centering.
        center_alignment=False,
        token_temperature=1.0,
        token_temperature_mode='text_length',
        token_temperature_reference_length=4.0,
        token_temperature_length_power=0.5,
        token_temperature_min=0.5,
        token_temperature_max=4.0,
        gate_temperature=1.0))
