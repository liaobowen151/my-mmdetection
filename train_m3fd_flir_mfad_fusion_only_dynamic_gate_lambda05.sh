#!/usr/bin/env bash

set -euo pipefail

cd /home/lbw/mmdetection

echo "[1/3] Training M3FD Fusion-only Dynamic Gate lambda=0.5"
PORT=29614 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/m3fd/textfusion_interest/fusion_only_dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/m3fd/multimodal/textfusion_interest50/fusion_only_dynamic_gate_lambda05/fusion_only_dynamic_gate_lambda05_train50

echo "[2/3] Training FLIR Fusion-only Dynamic Gate lambda=0.5"
PORT=29615 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/flir/textfusion_interest/fusion_only_dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/flir/multimodal/textfusion_interest50/fusion_only_dynamic_gate_lambda05/fusion_only_dynamic_gate_lambda05_train50

echo "[3/3] Training MFAD Fusion-only Dynamic Gate lambda=0.5"
PORT=29616 \
CUDA_VISIBLE_DEVICES=0,1 \
PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
bash tools/dist_train.sh \
configs/mfad/textfusion_interest/fusion_only_dynamic_gate_lambda05.py \
2 \
--work-dir \
workdir/mfad/multimodal/textfusion_interest50/fusion_only_dynamic_gate_lambda05/fusion_only_dynamic_gate_lambda05_train50

echo "All three Fusion-only experiments completed successfully."
