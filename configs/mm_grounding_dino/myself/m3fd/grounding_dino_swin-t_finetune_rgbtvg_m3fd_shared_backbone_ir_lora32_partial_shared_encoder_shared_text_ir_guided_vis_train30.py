"""M3FD private visual/fusion and shared text early layers (方案 B)."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_shared_encoder_'
    'independent_hungarian_ir_guided_vis_train30.py')

custom_imports = dict(
    imports=[
        'projects.myself_partial_shared_encoder_ir_guided_vis_'
        'mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='PartialSharedEncoderSharedTextIRGuidedVISGroundingDINO')
