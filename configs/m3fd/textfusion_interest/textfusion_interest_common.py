"""Common 50-epoch M3FD TextFusion-interest experiment settings."""

_base_ = '../../../workdir/m3fd/multimodal/shared_backbone_ir_lora32_ir2vis_deformable_add_ir/groundingdino_rgbtvg_m3fd_ir_lora32_ir2vis_deformable_add_ir_train30/grounding_dino_swin-t_finetune_rgbtvg_m3fd_shared_backbone_ir_lora32_ir2vis_deformable_add_ir_train30.py'

custom_imports = dict(
    imports=['projects.myself_ir_lora_textfusion_interest_mmgroundingdino'],
    allow_failed_imports=False)

model = dict(
    type='IRLoRATextFusionInterestGroundingDINO',
    data_preprocessor=dict(pad_seg=True, seg_pad_value=0),
    multimodal_fusion=dict(
        _delete_=True,
        type='TextFusionInterestFusion',
        fusion_mode='direct_text',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4,
        num_points=4,
        gate_hidden_dims=256,
        gate_init_bias=2.0))

train_pipeline = [
    dict(
        type='LoadPairedImagesFromFile',
        ir_dir='data/M3FD/ir/',
        backend_args=None),
    dict(type='LoadAnnotations', with_bbox=True, with_seg=True),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(0.01, 0.01)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=
        '/home/lbw/mmdetection/huggingface-hub/transformers/bert-base-uncased/',
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'vis_img_path', 'ir_img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode',
                   'interest_mask_sample_id'))
]

train_dataloader = dict(
    dataset=dict(
        type='M3FDInterestMaskODVGDataset',
        interest_mask_dir='mask/',
        pipeline=train_pipeline))

max_epochs = 50
train_cfg = dict(type='EpochBasedTrainLoop', max_epochs=50, val_interval=1)
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=50,
        by_epoch=True,
        milestones=[35, 46],
        gamma=0.1)
]

custom_hooks = [
    dict(type='TextFusionInterestScheduleHook', lambda_interest_max=0.5)
]

# Child configs replace these two fields and set an independent work_dir.
