"""FLIR dual-backbone baseline with concat-linear VIS/IR fusion."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_base.py'  # noqa: E501

custom_imports = dict(
    imports=['projects.myself_dual_backbone_simple_fusion_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    type='DualBackboneSimpleFusionGroundingDINO',
    multimodal_fusion=dict(type='ConcatLinearFusion', embed_dims=256))

max_epochs = 12
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[8, 11],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
