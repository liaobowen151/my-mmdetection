"""MFAD IR-LoRA32: shared-sampling IR-minus-VIS deformable fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_mfad_shared_backbone_ir_lora32_neck_add_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_ir2vis_deformable_difference_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRAIRToVISDeformableGroundingDINO',
    multimodal_fusion=dict(
        type='IRToVISDeformableDifferenceFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4))
