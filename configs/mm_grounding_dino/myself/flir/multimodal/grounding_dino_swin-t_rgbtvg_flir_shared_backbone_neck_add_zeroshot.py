"""FLIR zero-shot baseline with shared post-neck VIS/IR addition."""

_base_ = './grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_dual_backbone_full_finetune_base.py'

custom_imports = dict(
    imports=['projects.myself_multimodal_mmgroundingdino'],
    allow_failed_imports=False)

# There is exactly one official Swin backbone and one official ChannelMapper
# neck. No LoRA, second backbone, or learnable fusion module is introduced.
model = dict(
    type='SharedBackboneNeckAddZeroShotGroundingDINO',
    freeze_visual_backbone=True,
    freeze_language_model=True)

randomness = dict(seed=42)

# This configuration is evaluation-only. Disabling the training/validation
# loops also prevents accidentally treating this architecture as trained.
train_cfg = None
train_dataloader = None
optim_wrapper = None
param_scheduler = None
val_cfg = None
val_dataloader = None
val_evaluator = None

load_from = (
    'checkpoints/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_'
    '20231204_095047-b448804b.pth')

work_dir = (
    'workdir/flir/multimodal/shared_backbone_neck_add_zeroshot/'
    'groundingdino_rgbtvg_flir_shared_backbone_neck_add_zeroshot')
