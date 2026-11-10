#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

echo "[1/2] Training FLIR TextFusion dynamic gate lambda=0.5"
PORT=29595 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/flir/textfusion_interest/dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/flir/multimodal/textfusion_interest/dynamic_gate_lambda05_train50

echo "[2/2] Training MFAD TextFusion dynamic gate lambda=0.5"
PORT=29596 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mfad/textfusion_interest/dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/mfad/multimodal/textfusion_interest/dynamic_gate_lambda05_train50

echo "Both training jobs completed successfully."
