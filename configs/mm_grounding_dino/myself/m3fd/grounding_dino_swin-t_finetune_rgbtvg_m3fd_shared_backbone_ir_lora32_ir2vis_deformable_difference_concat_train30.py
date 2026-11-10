"""M3FD IR-LoRA32: joint [IR-VIS, IR, VIS] deformable fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_ir2vis_deformable_difference_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='IRLoRAIRToVISDeformableGroundingDINO',
    multimodal_fusion=dict(
        type='IRToVISDeformableDifferenceConcatFusion',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4))
