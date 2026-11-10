#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

run_pair() {
    local strategy="$1"
    local port_on="$2"
    local port_off="$3"
    local experiment_root="workdir/m3fd/multimodal/shared_backbone_ir_lora32_neck_add_referent_rerank_${strategy}_train50"
    local run_name="groundingdino_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_${strategy}_train50"
    local config="configs/mm_grounding_dino/myself/m3fd/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_neck_add_referent_rerank_${strategy}_train50.py"
    local checkpoint="${experiment_root}/${run_name}/epoch_50.pth"
    local on_dir="${experiment_root}/ablation_rerank_on_epoch50"
    local off_dir="${experiment_root}/ablation_rerank_off_epoch50"

    echo "Testing ${strategy} epoch 50 with reranking enabled..."
    PORT="${port_on}" \
    CUDA_VISIBLE_DEVICES=0,1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash tools/dist_test.sh \
    "${config}" \
    "${checkpoint}" \
    2 \
    --work-dir "${on_dir}" \
    --out "${on_dir}/predictions.pkl"

    echo "Testing ${strategy} epoch 50 with reranking disabled..."
    PORT="${port_off}" \
    CUDA_VISIBLE_DEVICES=0,1 \
    PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
    bash tools/dist_test.sh \
    "${config}" \
    "${checkpoint}" \
    2 \
    --work-dir "${off_dir}" \
    --out "${off_dir}/predictions.pkl" \
    --cfg-options model.bbox_head.rerank_score_weight=0.0
}

echo "[1/3] Running paired tests for random negatives..."
run_pair random 29630 29631

echo "[2/3] Running paired tests for high-score negatives..."
run_pair score 29632 29633

echo "[3/3] Running paired tests for deduplicated high-score negatives..."
run_pair diverse 29634 29635

echo "All paired epoch-50 tests completed successfully."
