_base_ = '../../grounding_dino_swin-t_pretrain_obj365.py'

custom_imports = dict(
    imports=['projects.myself_multimodal_mmgroundingdino'],
    allow_failed_imports=False)

randomness = dict(seed=42)

model = dict(
    type='MultimodalGroundingDINO',
    freeze_visual_backbone=True,
    freeze_language_model=True,
    multimodal_fusion=dict(
        type='MultimodalFusionBlock',
        embed_dims=256,
        num_heads=8,
        dropout=0.0,
        num_feature_levels=4),
    # LoadPairedImagesFromFile decodes both modalities in RGB order and
    # concatenates them as [VIS_RGB, IR_RGB]. Resize keeps the aspect ratio;
    # divisor=224 pads only the right/bottom to a fixed 224 x 224 tensor.
    data_preprocessor=dict(
        _delete_=True,
        type='PairedDetDataPreprocessor',
        mean=[
            123.675, 116.28, 103.53,
            123.675, 116.28, 103.53,
        ],
        std=[
            58.395, 57.12, 57.375,
            58.395, 57.12, 57.375,
        ],
        pad_mask=False,
        pad_size_divisor=224,
        pad_value=0))

data_root = 'data/MFAD/'
ir_dir = data_root + 'ir/'

train_pipeline = [
    dict(
        type='LoadPairedImagesFromFile',
        ir_dir=ir_dir,
        backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    # No flip or crop: REC expressions may contain spatial/directional words.
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'vis_img_path', 'ir_img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

test_pipeline = [
    dict(
        type='LoadPairedImagesFromFile',
        ir_dir=ir_dir,
        backend_args=_base_.backend_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'vis_img_path', 'ir_img_path',
                   'ori_shape', 'img_shape', 'scale_factor', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/rgbtvg_mfad_train_vg.json',
        # The paired loader derives data/MFAD/ir/<same filename> from VIS.
        data_prefix=dict(img='vis/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file='annotations/rgbtvg_mfad_val.json',
        data_prefix=dict(img='vis/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None))

test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(type='DefaultSampler', shuffle=False),
    dataset=dict(
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file='annotations/rgbtvg_mfad_test.json',
        data_prefix=dict(img='vis/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None))

val_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=data_root + 'annotations/rgbtvg_mfad_val.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

test_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=data_root + 'annotations/rgbtvg_mfad_test.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2))

max_epochs = 12
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[9],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        by_epoch=True,
        interval=1,
        max_keep_ckpts=3,
        save_best='refexp/refcoco_precision@1',
        rule='greater'))
load_from = (
    'checkpoints/'
    'grounding_dino_swin-t_pretrain_obj365_goldg_'
    'grit9m_v3det_20231204_095047-b448804b.pth')
