"""M3FD IR-LoRA32 post-neck addition with referent-only token positives."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_train50.py'

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/rgbtvg_m3fd_train_vg_referent.json'))

load_from = (
    'checkpoints/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_'
    'grit9m_v3det_20231204_095047-b448804b.pth')

work_dir = (
    'workdir/m3fd/multimodal/'
    'shared_backbone_ir_lora32_neck_add_referent_train50/'
    'groundingdino_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_neck_add_referent_train50')
