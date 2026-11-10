"""Fixed Last B1 using detached final encoder text instead of BERT text."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_subject_head_context_fixed_last_l10_train50.py'

model = dict(
    context_text_source='encoder',
    bbox_head=dict(context_text_dim=256))
work_dir = ('workdir/m3fd/multimodal/'
            'shared_backbone_ir_lora32_neck_add_subject_head_context_'
            'encoder_text_fixed_last_l10_train50')
