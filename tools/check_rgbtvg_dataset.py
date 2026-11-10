from mmengine.config import Config
from mmdet.registry import DATASETS
from mmdet.utils import register_all_modules


register_all_modules()

config = Config.fromfile(
    'configs/mm_grounding_dino/refcoco/'
    'grounding_dino_swin-t_finetune_8xb4_5e_refcoco.py'
)

train_dataset_cfg = config.train_dataloader.dataset

train_dataset_cfg.data_root = 'data/FLIR'
train_dataset_cfg.ann_file = (
    'annotations/'
    'rgbtvg_flir_train_vg.json'
)
train_dataset_cfg.data_prefix = dict(
    img='vis/'
)

dataset = DATASETS.build(train_dataset_cfg)

print('Dataset length:', len(dataset))

sample = dataset[0]

print('Sample keys:', sample.keys())
print('Input shape:', sample['inputs'].shape)
print('Data sample:', sample['data_samples'])
print('Text:', sample['data_samples'].text)
print(
    'Tokens positive:',
    sample['data_samples'].tokens_positive
)
print(
    'GT boxes:',
    sample['data_samples'].gt_instances.bboxes
)