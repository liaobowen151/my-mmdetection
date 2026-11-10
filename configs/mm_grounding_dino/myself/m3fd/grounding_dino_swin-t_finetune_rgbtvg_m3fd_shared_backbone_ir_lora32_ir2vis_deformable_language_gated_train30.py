"""M3FD IR-LoRA32 with RIS-FUSION-style language-gated fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_ir2vis_deformable_add_ir_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_ir2vis_deformable_language_gated_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRAIRToVISDeformableLanguageGatedGroundingDINO',
    multimodal_fusion=dict(
        type='IRToVISDeformableLanguageGatedFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        language_num_heads=4,
        language_hidden_dims=128,
        language_levels=(2, 3),
        gamma_scale=0.1,
        text_temperature=1.0,
        token_pooling_temperature=0.25,
        modality_temperature=0.25,
        spatial_temperature=0.25,
        text_preserve_loss_weight=0.5,
        exclude_boundary_tokens=True))
