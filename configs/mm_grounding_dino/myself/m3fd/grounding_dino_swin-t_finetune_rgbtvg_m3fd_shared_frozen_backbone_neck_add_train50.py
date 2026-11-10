"""M3FD frozen shared Swin/LM with post-neck VIS+IR addition."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_multimodal_dual_backbone_full_finetune_base.py'

custom_imports = dict(
    imports=['projects.myself_multimodal_mmgroundingdino'],
    allow_failed_imports=False)

# One shared frozen Swin and one shared ChannelMapper process both modalities.
# There is no LoRA, second backbone, or learnable fusion module.
model = dict(
    type='SharedBackboneNeckAddZeroShotGroundingDINO',
    freeze_visual_backbone=True,
    freeze_language_model=True)

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))

max_epochs = 50

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[30, 45],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)
resume = False

work_dir = 'workdir/m3fd/multimodal/shared_frozen_backbone_neck_add_train50/groundingdino_rgbtvg_m3fd_shared_frozen_backbone_neck_add_train50'
