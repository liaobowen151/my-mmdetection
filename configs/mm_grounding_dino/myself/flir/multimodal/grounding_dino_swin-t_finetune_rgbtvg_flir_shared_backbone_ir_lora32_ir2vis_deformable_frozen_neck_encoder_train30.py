_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_shared_backbone_ir_lora32_ir2vis_deformable_train30.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_frozen_neck_encoder_mmgroundingdino',
    ],
    allow_failed_imports=False)

model = dict(type=(
    'FrozenNeckEncoderIRLoRAIRToVISDeformableGroundingDINO'))

work_dir = (
    'workdir/flir/multimodal/'
    'shared_backbone_ir_lora32_ir2vis_deformable_frozen_neck_encoder/'
    'groundingdino_rgbtvg_flir_ir_lora32_ir2vis_deformable_'
    'frozen_neck_encoder_train30')
