#!/usr/bin/env bash
set -euo pipefail

cd /home/lbw/mmdetection

export CUDA_VISIBLE_DEVICES=0,1
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

PORT=29622 bash tools/dist_train.sh \
  configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_subject_head_context_encoder_text_fixed_last_l10_train50.py \
  2 \
  --work-dir \
  workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_subject_head_context_encoder_text_fixed_last_l10_train50

PORT=29623 bash tools/dist_train.sh \
  configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_subject_head_context_encoder_text_fixed_all6_l10_train50.py \
  2 \
  --work-dir \
  workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_subject_head_context_encoder_text_fixed_all6_l10_train50
