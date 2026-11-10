"""Common A2 candidate-reranking setup for the M3FD referent model."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_train50.py'

custom_imports = dict(
    imports=['projects.myself_ir_lora_addition_rerank_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    bbox_head=dict(
        type='RerankGroundingDINOHead',
        rerank_strategy='random',
        num_rerank_negatives=9,
        negative_iou_threshold=0.3,
        diverse_nms_iou_threshold=0.7,
        rerank_loss_weight=1.0,
        rerank_temperature=1.0,
        rerank_score_weight=1.0,
        rerank_topk=10))
