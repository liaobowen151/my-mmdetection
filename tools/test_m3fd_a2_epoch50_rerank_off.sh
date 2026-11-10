#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

echo "[1/3] Testing random-negative epoch 50 with reranking disabled..."
PORT=29621 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_test.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50.py \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50/epoch_50.pth \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50/ablation_rerank_off_epoch50 \
--cfg-options \
model.bbox_head.rerank_score_weight=0.0

echo "[2/3] Testing high-score-negative epoch 50 with reranking disabled..."
PORT=29622 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_test.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50.py \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50/epoch_50.pth \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50/ablation_rerank_off_epoch50 \
--cfg-options \
model.bbox_head.rerank_score_weight=0.0

echo "[3/3] Testing deduplicated high-score-negative epoch 50 with reranking disabled..."
PORT=29623 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_test.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50.py \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50/epoch_50.pth \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50/ablation_rerank_off_epoch50 \
--cfg-options \
model.bbox_head.rerank_score_weight=0.0

echo "All three epoch-50 rerank-off tests completed successfully."
