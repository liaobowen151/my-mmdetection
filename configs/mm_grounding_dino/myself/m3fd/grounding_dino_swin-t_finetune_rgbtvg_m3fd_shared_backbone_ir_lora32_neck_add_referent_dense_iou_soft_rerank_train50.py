"""M3FD referent model with dense IoU-aware soft reranking."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_neck_add_referent_train50.py')

custom_imports = dict(
    imports=['projects.myself_ir_lora_addition_dense_rerank_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='DenseIoUSoftRerankGroundingDINOHead',
        positive_iou_threshold=0.5,
        negative_iou_threshold=0.3,
        iou_target_temperature=0.1,
        rerank_loss_weight=1.0,
        rerank_temperature=1.0,
        rerank_score_weight=1.0))

work_dir = (
    'workdir/m3fd/multimodal/'
    'shared_backbone_ir_lora32_neck_add_referent_dense_iou_soft_rerank_'
    'train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_'
    'add_referent_dense_iou_soft_rerank_train50')
