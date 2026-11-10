#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

echo "[1/3] Training A2 with random natural negatives..."
PORT=29605 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50.py \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_random_train50

echo "[2/3] Training A2 with high-score referent negatives..."
PORT=29606 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50.py \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_score_train50

echo "[3/3] Training A2 with deduplicated high-score negatives..."
PORT=29607 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50.py \
2 \
--work-dir \
workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50/groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_diverse_train50

echo "All three A2 reranking experiments completed successfully."
