"""M3FD shared encoder with independent IR/VIS Hungarian assignments."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_ir_guided_vis_train30.py')

custom_imports = dict(
    imports=[
        'projects.'
        'myself_shared_encoder_independent_hungarian_ir_guided_vis_'
        'mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='SharedEncoderIndependentHungarianIRGuidedVISGroundingDINO',
    bbox_head=dict(
        type='SharedEncoderIndependentHungarianIRGuidedVISGroundingDINOHead'))

custom_hooks = [dict(type='EpochIRGuidedMetricsHook')]
