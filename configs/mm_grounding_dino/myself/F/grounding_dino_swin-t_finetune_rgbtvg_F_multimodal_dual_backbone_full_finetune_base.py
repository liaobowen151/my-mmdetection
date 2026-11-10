"""F dataset dual-backbone baseline with direct VIS/IR addition."""

_base_ = (
    '../flir/multimodal/'
    'grounding_dino_swin-t_finetune_rgbtvg_flir_multimodal_'
    'dual_backbone_full_finetune_base.py')

data_root = 'data/F/'
ir_dir = data_root + 'ir/'

train_pipeline = [
    dict(
        type='LoadPairedImagesFromFile',
        ir_dir=ir_dir,
        backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
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
    batch_size=4,
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/annotations_refexp_train_vg.json',
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
        ann_file='annotations/annotations_refexp_val.json',
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
        ann_file='annotations/annotations_refexp_test.json',
        data_prefix=dict(img='vis/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None))

val_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=data_root + 'annotations/annotations_refexp_val.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

test_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=data_root + 'annotations/annotations_refexp_test.json',
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10))

work_dir = (
    'workdir/F/multimodal/dual_backbone_base/'
    'groundingdino_rgbtvg_F_multimodal_'
    'dual_backbone_full_finetune_base_train50')
