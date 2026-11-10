"""MFAD shared-backbone IR-LoRA-32 neck addition trained for 50 epochs."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_mfad_shared_backbone_ir_lora32_neck_add_train30.py'

max_epochs = 50

param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[35, 46],
        gamma=0.1)
]

train_cfg = dict(max_epochs=max_epochs, val_interval=1)

work_dir = 'workdir/mfad/multimodal/shared_backbone_ir_lora32_neck_add/groundingdino_rgbtvg_mfad_shared_backbone_ir_lora32_neck_add_train50'
