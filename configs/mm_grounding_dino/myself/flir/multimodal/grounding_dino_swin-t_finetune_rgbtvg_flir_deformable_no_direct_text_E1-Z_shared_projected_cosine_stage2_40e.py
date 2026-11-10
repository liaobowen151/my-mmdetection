"""Stage-2 40-epoch projected-cosine E1-Z experiment on FLIR.

This is a weights-only continuation from the sum-fusion epoch-10 checkpoint.
Optimizer and scheduler state intentionally start fresh.  The original global
milestones 30/45 become local stage-2 milestones 20/35.
"""

_base_ = [
    '../../../../../workdir/flir/multimodal/'
    'dual_backbone_deformable_cross_attentionv2_no_direct_text_attention_'
    'E1-Z_hard_switch_epoch11/'
    'groundingdino_rgbtvg_flir_no_direct_text_deformable_E1-Z_'
    'hard_switch_epoch11_train50/'
    'grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_'
    'full_finetune_deformable_cross_attentionv2_no_direct_text_attention_'
    'E1-Z_hard_switch_epoch11.py'
]

custom_imports = dict(
    imports=[
        'projects.myself_dual_backbone_deformable_spatial_expectation_'
        'projected_cosine_mmgroundingdino',
    ],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneNoDirectTextDeformableSpatialExpectationGroundingDINO',
    multimodal_fusion=dict(
        type=('NoDirectTextDeformableProjectedCosineSpatialExpectation'
              'FusionBlock'),
        similarity_temperature=1.0,
        similarity_normalization_eps=1e-6))

# E1-Z is active from the first epoch of this second stage.
custom_hooks = [
    dict(type='SpatialChannelExpectationStatisticsHook'),
]

train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=40, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=40,
        by_epoch=True,
        milestones=[20, 35],
        gamma=0.1),
]

load_from = ('/home/lbw/mmdetection/workdir/flir/multimodal/base10weight/'
             'epoch_10.pth')
resume = False
launcher = 'none'

work_dir = (
    'workdir/flir/multimodal/'
    'dual_backbone_deformable_cross_attentionv2_no_direct_text_attention_'
    'E1-Z_shared_projected_cosine_stage2_40e/'
    'groundingdino_rgbtvg_flir_deformable_no_direct_text_E1-Z_'
    'shared_projected_cosine_stage2_40e')
