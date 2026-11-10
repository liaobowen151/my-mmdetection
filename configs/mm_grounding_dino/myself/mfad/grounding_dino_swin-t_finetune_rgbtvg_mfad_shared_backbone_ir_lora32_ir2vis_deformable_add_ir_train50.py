"""MFAD IR-LoRA32 IR-to-VIS deformable add-IR trained for 50 epochs."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_mfad_shared_backbone_ir_lora32_ir2vis_deformable_add_ir_train30.py'

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

work_dir = 'workdir/mfad/multimodal/shared_backbone_ir_lora32_ir2vis_deformable_add_ir_train50/groundingdino_rgbtvg_mfad_ir_lora32_ir2vis_deformable_add_ir_train50'
