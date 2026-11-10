#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

echo "[1/2] Training FLIR Bidirectional Deformable Dynamic Gate lambda=0.5"
PORT=29619 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/flir/textfusion_interest/bidirectional_deformable_dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/flir/multimodal/textfusion_interest50/bidirectional_deformable_dynamic_gate_lambda05/bidirectional_deformable_dynamic_gate_lambda05_train50

echo "[2/2] Training MFAD Bidirectional Deformable Dynamic Gate lambda=0.5"
PORT=29620 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mfad/textfusion_interest/bidirectional_deformable_dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/mfad/multimodal/textfusion_interest50/bidirectional_deformable_dynamic_gate_lambda05/bidirectional_deformable_dynamic_gate_lambda05_train50

echo "Both bidirectional deformable dynamic-gate experiments completed successfully."
