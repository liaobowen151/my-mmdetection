"""FLIR: frozen shared Swin, IR-only LoRA-32 and post-neck addition."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_base.py'

custom_imports = dict(
    imports=['projects.myself_ir_lora_addition_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    type='IRLoRAAdditionGroundingDINO',
    freeze_visual_backbone=True,
    backbone=dict(
        type='IRLoRASwinTransformer',
        lora_rank=32,
        lora_dropout=0.1,
        with_cp=False))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))

max_epochs = 30
param_scheduler = [
    dict(
        type='MultiStepLR', begin=0, end=max_epochs, by_epoch=True,
        milestones=[21, 28], gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
