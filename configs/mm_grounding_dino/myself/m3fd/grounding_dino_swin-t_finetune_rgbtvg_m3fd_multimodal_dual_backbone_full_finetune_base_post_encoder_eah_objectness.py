"""M3FD post-encoder EAH Top-K with proposal-level EAM supervision."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_'
    'dual_backbone_full_finetune_base_post_encoder_eah.py')

custom_imports = dict(
    imports=[
        'projects.'
        'myself_dual_backbone_post_encoder_eah_objectness_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackbonePostEncoderEAHObjectnessGroundingDINO',
    bbox_head=dict(
        type='PostEncoderEAHObjectnessGroundingDINOHead',
        loss_eam=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5)))

# Match the launcher recorded in the referenced two-GPU training config.
launcher = 'pytorch'

work_dir = (
    'workdir/m3fd/multimodal/'
    'dual_backbone_base_post_encoder_eah_objectness/'
    'groundingdino_rgbtvg_m3fd_multimodal_dual_backbone_'
    'full_finetune_base_post_encoder_eah_objectness_train')
