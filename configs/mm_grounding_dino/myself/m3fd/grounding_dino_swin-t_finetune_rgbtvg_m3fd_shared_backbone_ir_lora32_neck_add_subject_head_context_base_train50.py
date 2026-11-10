"""Common M3FD subject-head proposal and context-reranking experiment."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_train50.py'

custom_imports = dict(
    imports=[
        'projects.myself_ir_lora_subject_head_context_mmgroundingdino'
    ],
    allow_failed_imports=False)

model = dict(
    type='SubjectHeadContextGroundingDINO',
    subject_head_classifier_checkpoint=(
        '/home/lbw/mmdetection/work_dirs/subject_head_classifier/'
        'combined_v1/run1/subject_head_classifier_best.pth'),
    subject_head_hidden_dim=256,
    subject_head_dropout=0.1,
    bbox_head=dict(
        type='SubjectHeadContextGroundingDINOHead',
        context_fusion='fixed',
        b1_supervision='last',
        b1_loss_weight=1.0,
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        iou_target_temperature=0.1,
        importance_temperature=1.0,
        evidence_temperature=1.0,
        rerank_temperature=1.0,
        score_standardize_eps=1e-6,
        score_standardize_min_std=0.05,
        context_text_dim=768,
        match_dim=256,
        context_hidden_dim=256,
        context_dropout=0.1))

train_dataloader = dict(
    dataset=dict(
        ann_file='annotations/rgbtvg_m3fd_train_vg_subject_head.json'))

