"""M3FD ODVG dataset with one target-specific interest mask per row."""

import os.path as osp

from mmdet.datasets.odvg import ODVGDataset
from mmdet.registry import DATASETS


@DATASETS.register_module()
class InterestMaskODVGDataset(ODVGDataset):
    """Attach ``mask/{sample_id:06d}_{image_stem}.png`` to each sample.

    M3FD REC annotations can contain several descriptions for the same image,
    so the row/sample id is deliberately part of the mask filename.
    """

    def __init__(self, *args, interest_mask_dir: str = 'mask/', **kwargs):
        self.interest_mask_dir = interest_mask_dir
        super().__init__(*args, **kwargs)

    def load_data_list(self):
        data_list = super().load_data_list()
        for sample_id, data_info in enumerate(data_list):
            basename = osp.basename(data_info['img_path'])
            stem, _ = osp.splitext(basename)
            mask_name = f'{sample_id:06d}_{stem}.png'
            data_info['seg_map_path'] = osp.join(self.data_root,
                                                 self.interest_mask_dir,
                                                 mask_name)
            data_info['interest_mask_sample_id'] = sample_id
        return data_list


@DATASETS.register_module()
class M3FDInterestMaskODVGDataset(InterestMaskODVGDataset):
    """Backward-compatible name used by the original M3FD configs."""

    pass
