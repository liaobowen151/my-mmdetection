"""M3FD IR-LoRA32 post-neck addition trained for 50 epochs."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_train30.py'

max_epochs = 50

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))

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
