_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_subject_head_context_base_train50.py'

model = dict(bbox_head=dict(
    context_fusion='fixed', b1_supervision='last', b1_loss_weight=0.5))
work_dir = ('workdir/m3fd/multimodal/'
            'shared_backbone_ir_lora32_neck_add_subject_head_context_'
            'fixed_last_l05_train50')

