"""M3FD IR-LoRA32 with language-gated fusion at all four levels."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_ir2vis_deformable_language_gated_train30.py'

model = dict(
    multimodal_fusion=dict(
        language_levels=(0, 1, 2, 3)))
