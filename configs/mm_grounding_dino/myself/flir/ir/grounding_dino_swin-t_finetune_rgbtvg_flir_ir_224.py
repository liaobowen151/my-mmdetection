_base_ = '../grounding_dino_swin-t_pretrain_obj365.py'

model = dict(
    # Resize keeps the aspect ratio and limits both sides to <= 224. Setting
    # the divisor to 224 then pads only the right/bottom sides to 224 x 224.
    data_preprocessor=dict(
        type='DetDataPreprocessor',
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_mask=False,
        pad_size_divisor=224,
        pad_value=0))

data_root = 'data/FLIR/'
train_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='RandomFlip', prob=0.0),
    dict(
        type='Resize',
        scale=(224, 224),
        keep_ratio=True),
    dict(type='FilterAnnotations', min_gt_bbox_wh=(1e-2, 1e-2)),
    dict(
        type='RandomSamplingNegPos',
        tokenizer_name=_base_.lang_model_name,
        num_sample_negative=85,
        max_tokens=256),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'flip', 'flip_direction', 'text',
                   'custom_entities', 'tokens_positive', 'dataset_mode'))
]

test_pipeline = [
    dict(type='LoadImageFromFile', backend_args=_base_.backend_args),
    dict(type='Resize', scale=(224, 224), keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor', 'text', 'custom_entities',
                   'tokens_positive'))
]

train_dataloader = dict(
    dataset=dict(
        _delete_=True,
        type='ODVGDataset',
        data_root=data_root,
        ann_file='annotations/rgbtvg_flir_train_vg.json',
        data_prefix=dict(img='ir/'),
        filter_cfg=dict(filter_empty_gt=False, min_size=32),
        return_classes=True,
        pipeline=train_pipeline))

# -------------------------------------------------#

val_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    dataset=dict(
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file='annotations/rgbtvg_flir_val.json',
        data_prefix=dict(
            img='ir/'
        ),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)


test_dataloader = dict(
    batch_size=1,
    num_workers=2,
    persistent_workers=True,
    drop_last=False,
    sampler=dict(
        type='DefaultSampler',
        shuffle=False
    ),
    dataset=dict(
        type='MDETRStyleRefCocoDataset',
        data_root=data_root,
        ann_file=('annotations/rgbtvg_flir_test.json'),
        data_prefix=dict(img='ir/'),
        test_mode=True,
        return_classes=True,
        pipeline=test_pipeline,
        backend_args=None
    )
)

val_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=(data_root + 'annotations/rgbtvg_flir_val.json'),
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10)
)

test_evaluator = dict(
    _delete_=True,
    type='RefExpMetric',
    ann_file=(data_root + 'annotations/rgbtvg_flir_test.json'),
    metric='bbox',
    iou_thrs=0.5,
    topk=(1, 5, 10)
)



optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=0.0002, weight_decay=0.0001),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    paramwise_cfg=dict(
        custom_keys={
            'backbone': dict(lr_mult=0.1),
        }))

# learning policy
max_epochs = 5
param_scheduler = [
    dict(
        type='MultiStepLR',
        begin=0,
        end=max_epochs,
        by_epoch=True,
        milestones=[3],
        gamma=0.1)
]
train_cfg = dict(max_epochs=max_epochs, val_interval=1)

load_from = ('checkpoints/grounding_dino_swin-t_pretrain_obj365_goldg_grit9m_v3det_20231204_095047-b448804b.pth')
