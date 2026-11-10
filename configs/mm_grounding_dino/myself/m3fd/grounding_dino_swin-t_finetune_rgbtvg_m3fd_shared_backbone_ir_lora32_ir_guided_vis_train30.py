"""M3FD IR-guided proposal selection with VIS-only decoding."""

_base_ = (
    './grounding_dino_swin-t_finetune_rgbtvg_m3fd_'
    'shared_backbone_ir_lora32_neck_add_train30.py')

custom_imports = dict(
    imports=['projects.myself_ir_guided_vis_mmgroundingdino'],
    allow_failed_imports=False)

# The shared encoder is checkpointed and executed twice (VIS and IR). DDP's
# static-graph mode supports the resulting repeated reentrant backward hooks.
model_wrapper_cfg = dict(
    type='MMDistributedDataParallel',
    find_unused_parameters=False,
    static_graph=True)

model = dict(
    type='IRGuidedVISGroundingDINO',
    bbox_head=dict(
        type='IRGuidedVISGroundingDINOHead',
        encoder_loss_weights=dict(
            ir_cls=0.5,
            ir_bbox=1.0,
            ir_iou=1.0,
            vis_cls=1.0,
            vis_bbox=0.25,
            vis_iou=0.25)))
