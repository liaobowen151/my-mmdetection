"""M3FD dual-encoder IR-guided proposal selection with VIS decoding."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_ir_guided_vis_train30.py')

custom_imports = dict(
    imports=['projects.myself_dual_encoder_ir_guided_vis_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    type='DualEncoderIRGuidedVISGroundingDINO',
    bbox_head=dict(type='DualEncoderIRGuidedVISGroundingDINOHead'))

custom_hooks = [dict(type='EpochIRGuidedMetricsHook')]
