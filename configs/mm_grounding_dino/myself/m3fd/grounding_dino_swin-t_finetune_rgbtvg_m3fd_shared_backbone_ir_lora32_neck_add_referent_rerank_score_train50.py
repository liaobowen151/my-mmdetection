"""A2-Score: listwise reranking with highest-anchor negatives."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_common_train50.py'

model = dict(bbox_head=dict(rerank_strategy='score'))

work_dir = (
    'workdir/m3fd/multimodal/'
    'shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50/'
    'groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_'
    'referent_rerank_score_train50')
