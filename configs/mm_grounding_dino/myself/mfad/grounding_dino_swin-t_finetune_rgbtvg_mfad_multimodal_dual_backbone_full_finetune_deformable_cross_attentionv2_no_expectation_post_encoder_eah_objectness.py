"""MFAD: deformable no-expectation fusion plus EAH objectness."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_mfad_multimodal_'
    'dual_backbone_full_finetune_deformable_cross_attentionv2.py')

custom_imports = dict(
    imports=[
        'projects.'
        'myself_dual_backbone_deformable_no_expectation_'
        'post_encoder_eah_objectness_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type=(
        'DualBackboneDeformableNoExpectationPostEncoderEAHObjectness'
        'GroundingDINO'),
    eah_exclude_cls_token=True,
    bbox_head=dict(
        type='PostEncoderEAHObjectnessGroundingDINOHead',
        loss_eam=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=0.5)))

work_dir = (
    'workdir/mfad/multimodal/'
    'dual_backbone_deformable_cross_attentionv2_no_expectation_'
    'post_encoder_eah_objectness/'
    'groundingdino_rgbtvg_mfad_multimodal_dual_backbone_'
    'deformable_cross_attentionv2_no_expectation_'
    'post_encoder_eah_objectness_train')
