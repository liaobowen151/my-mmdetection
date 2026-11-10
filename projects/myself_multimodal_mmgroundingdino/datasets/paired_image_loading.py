import os.path as osp
from typing import Optional

import mmcv
import numpy as np
from mmcv.transforms import BaseTransform
from mmengine.fileio import get

from mmdet.registry import TRANSFORMS


@TRANSFORMS.register_module()
class LoadPairedImagesFromFile(BaseTransform):
    """Load aligned visible and infrared images as one six-channel image.

    The dataset still provides ``img_path`` for the visible image. The
    infrared image is resolved by joining ``ir_dir`` with the visible image's
    basename. Both images are decoded in RGB order and concatenated along the
    channel dimension as ``[VIS_RGB, IR_RGB]``.

    Args:
        ir_dir (str): Directory containing infrared images whose filenames
            match the visible images.
        to_float32 (bool): Convert the concatenated image to float32.
        backend_args (dict, optional): File backend arguments.
        imdecode_backend (str): Backend used by ``mmcv.imfrombytes``.
    """

    def __init__(self,
                 ir_dir: str,
                 to_float32: bool = False,
                 backend_args: Optional[dict] = None,
                 imdecode_backend: str = 'cv2') -> None:
        self.ir_dir = ir_dir
        self.to_float32 = to_float32
        self.backend_args = backend_args
        self.imdecode_backend = imdecode_backend

    def transform(self, results: dict) -> dict:
        vis_img_path = results['img_path']
        ir_img_path = osp.join(self.ir_dir, osp.basename(vis_img_path))

        try:
            vis_bytes = get(vis_img_path, backend_args=self.backend_args)
        except Exception as exc:
            raise FileNotFoundError(
                f'Failed to load visible image: {vis_img_path}') from exc
        try:
            ir_bytes = get(ir_img_path, backend_args=self.backend_args)
        except Exception as exc:
            raise FileNotFoundError(
                f'Failed to load infrared image paired with '
                f'{vis_img_path}: {ir_img_path}') from exc

        vis_img = mmcv.imfrombytes(
            vis_bytes,
            flag='color',
            channel_order='rgb',
            backend=self.imdecode_backend)
        ir_img = mmcv.imfrombytes(
            ir_bytes,
            flag='color',
            channel_order='rgb',
            backend=self.imdecode_backend)

        if vis_img is None:
            raise ValueError(f'Failed to decode visible image: {vis_img_path}')
        if ir_img is None:
            raise ValueError(f'Failed to decode infrared image: {ir_img_path}')
        if vis_img.shape[:2] != ir_img.shape[:2]:
            raise ValueError(
                'Visible and infrared images must be spatially aligned, but '
                f'got VIS {vis_img.shape[:2]} and IR {ir_img.shape[:2]} for '
                f'{vis_img_path} and {ir_img_path}.')

        img = np.concatenate([vis_img, ir_img], axis=-1)
        if self.to_float32:
            img = img.astype(np.float32)

        results['img'] = img
        results['img_path'] = vis_img_path
        results['vis_img_path'] = vis_img_path
        results['ir_img_path'] = ir_img_path
        results['img_shape'] = img.shape[:2]
        results['ori_shape'] = img.shape[:2]
        return results

    def __repr__(self) -> str:
        return (f'{self.__class__.__name__}('
                f'ir_dir={self.ir_dir!r}, '
                f'to_float32={self.to_float32}, '
                f'imdecode_backend={self.imdecode_backend!r})')

