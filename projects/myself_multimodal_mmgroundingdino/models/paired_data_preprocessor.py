from numbers import Number
from typing import Optional, Sequence, Union

import torch

from mmdet.models.data_preprocessors import DetDataPreprocessor
from mmdet.registry import MODELS


@MODELS.register_module()
class PairedDetDataPreprocessor(DetDataPreprocessor):
    """Detection preprocessor for a concatenated VIS-RGB/IR-RGB tensor.

    MMEngine's image preprocessor intentionally accepts only one or three
    normalization values. This subclass keeps all official detection padding,
    metadata and batch-augmentation behavior, but registers six normalization
    values for inputs laid out as ``[VIS_RGB, IR_RGB]``.
    """

    def __init__(
        self,
        mean: Sequence[Number],
        std: Sequence[Number],
        pad_size_divisor: int = 1,
        pad_value: Union[float, int] = 0,
        pad_mask: bool = False,
        mask_pad_value: int = 0,
        pad_seg: bool = False,
        seg_pad_value: int = 255,
        boxtype2tensor: bool = True,
        non_blocking: Optional[bool] = False,
        batch_augments: Optional[list] = None,
    ) -> None:
        if len(mean) != 6 or len(std) != 6:
            raise ValueError(
                'PairedDetDataPreprocessor expects six mean/std values in '
                '[VIS_RGB, IR_RGB] order.')

        # Disable the parent's 1/3-channel normalization setup, then register
        # equivalent six-channel buffers. Its forward implementation can
        # otherwise normalize, pad, stack and populate metadata unchanged.
        super().__init__(
            mean=None,
            std=None,
            pad_size_divisor=pad_size_divisor,
            pad_value=pad_value,
            pad_mask=pad_mask,
            mask_pad_value=mask_pad_value,
            pad_seg=pad_seg,
            seg_pad_value=seg_pad_value,
            bgr_to_rgb=False,
            rgb_to_bgr=False,
            boxtype2tensor=boxtype2tensor,
            non_blocking=non_blocking,
            batch_augments=batch_augments)

        self._enable_normalize = True
        self.register_buffer(
            'mean', torch.tensor(mean).view(6, 1, 1), persistent=False)
        self.register_buffer(
            'std', torch.tensor(std).view(6, 1, 1), persistent=False)

