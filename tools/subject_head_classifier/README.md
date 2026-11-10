# Subject head classifier

This module trains only a small token-scoring classifier on frozen
GroundingDINO BERT features. Final checkpoints contain the classifier
`state_dict` only and never contain BERT parameters.

## Prepare and cache features

```bash
conda run -n openmmlab python tools/subject_head_classifier/prepare.py \
  --output-dir work_dirs/subject_head_classifier/features
```

The split is grouped by `filename`, so descriptions of one image cannot leak
between train, validation and test sets. Preparation also writes an annotation
audit and stops if any character span cannot be aligned.

## Train

```bash
conda run -n openmmlab python tools/subject_head_classifier/train.py \
  --feature-dir work_dirs/subject_head_classifier/features \
  --output-dir work_dirs/subject_head_classifier/run1
```

The deployable file is `subject_head_classifier_best.pth`. It is a plain
classifier `state_dict`, with no BERT weights.

## Combined M3FD + FLIR + MFAD training

```bash
conda run -n openmmlab python tools/subject_head_classifier/prepare_combined.py \
  --output-dir work_dirs/subject_head_classifier/combined_v1/features

conda run -n openmmlab python tools/subject_head_classifier/train.py \
  --feature-dir work_dirs/subject_head_classifier/combined_v1/features \
  --output-dir work_dirs/subject_head_classifier/combined_v1/run1 \
  --sampling sqrt --batch-size 128 --epochs 60 --patience 10
```

Combined splitting keeps both identical normalized captions and all captions
from one source image in a single split. The best checkpoint is selected by
the macro validation exact-match accuracy over the three datasets.

## Predict

```bash
conda run -n openmmlab python tools/subject_head_classifier/predict.py \
  'White motorcycle with a rider near car' \
  --checkpoint work_dirs/subject_head_classifier/run1/subject_head_classifier_best.pth
```
