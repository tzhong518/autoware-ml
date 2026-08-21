"""Camera image masking transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import cv2
import numpy as np
import numpy.typing as npt

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.utils import (
    as_hwc_image_list,
    is_chw_image,
    mask_polygon_to_pixels,
    normalize_mask_polygon,
    restore_image_container,
)


class GridMask(BaseTransform):
    """Apply grid masking augmentation to one image or a list of images."""

    _required_keys = ["img"]

    def __init__(self, *, p: float = 0.7, ratio: float = 0.5, rotate: int = 1) -> None:
        """Initialize the GridMask transform.

        Args:
            p: Probability of applying the transform.
            ratio: Fraction of each grid period that is masked out.
            rotate: Maximum absolute rotation in degrees applied to the mask.
        """
        self.p = p
        self.ratio = ratio
        self.rotate = rotate

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Mask images with a regular grid pattern."""
        images, format_info = as_hwc_image_list(input_dict["img"])
        masked = [self._grid_mask(image) for image in images]
        input_dict["img"] = restore_image_container(input_dict["img"], masked, format_info)
        return input_dict

    def _grid_mask(self, image: npt.NDArray) -> npt.NDArray:
        """Apply the grid mask to a single image."""
        height, width = image.shape[:2]
        period = np.random.randint(32, max(33, min(height, width)))
        cut = max(1, int(period * self.ratio))
        mask = np.ones((height, width), dtype=np.float32)
        offset_x = np.random.randint(period)
        offset_y = np.random.randint(period)
        for x in range(offset_x, width, period):
            mask[:, x : x + cut] = 0
        for y in range(offset_y, height, period):
            mask[y : y + cut, :] = 0
        if self.rotate > 0:
            angle = np.random.uniform(-self.rotate, self.rotate)
            rotation = cv2.getRotationMatrix2D((width / 2, height / 2), angle, 1.0)
            mask = cv2.warpAffine(mask, rotation, (width, height))
        if image.ndim == 3:
            mask = mask[..., None]
        return (image.astype(np.float32) * mask).astype(image.dtype)


class EgoAreaMaskPaint(BaseTransform):
    """Paint each camera's ego-occluded region a flat color in the image.

    Cameras mounted on the ego vehicle can have part of their field of view
    occluded by the vehicle's own body or mirrors. This paints that fixed,
    per-camera normalized polygon region a flat color (black by default)
    directly onto the loaded image, independent of any label filtering.

    Required keys:
        img: Per-camera images, any of the containers accepted by
            ``as_hwc_image_list`` (stacked ``(num_cams, C, H, W)`` array,
            list of per-camera arrays, or a single image).
        camera_names: Per-sample camera names in the same order as the
            per-camera entries in ``img``. Read from the sample rather than
            from config because ``LoadMultiViewImagesFromFiles(shuffle_order=True)``
            permutes the camera order per sample; indexing a static config
            list would paint each polygon onto the wrong camera.

    Generated keys:
        img: Same container as the input, with each configured camera's
            polygon region painted over.
    """

    _required_keys = ["img", "camera_names"]

    def __init__(
        self,
        *,
        camera_masks: dict[str, Sequence[float]],
        mask_color: Sequence[float] = (0, 0, 0),
    ) -> None:
        """Initialize the EgoAreaMaskPaint transform.

        Args:
            camera_masks: Mapping from camera name to a flat normalized
                ``[x0, y0, x1, y1, ...]`` polygon marking that camera's
                ego-occluded region. Cameras absent from this mapping are left untouched.
            mask_color: Fill color applied inside each polygon, in the same
                channel order as the loaded images.
        """
        self.camera_polygons = {
            str(camera): normalize_mask_polygon(polygon) for camera, polygon in camera_masks.items()
        }
        self.mask_color = tuple(mask_color)
        # (camera, height, width) -> row indices, column indices of masked pixels.
        # The polygons and the image size are fixed for a run, so the rasterized
        # mask is computed once per camera instead of once per sample.
        self._index_cache: dict[tuple[str, int, int], tuple[npt.NDArray, npt.NDArray]] = {}

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Paint the configured ego-mask polygon onto each camera's image.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary with masked regions painted.
        """
        camera_names = list(input_dict["camera_names"])
        images = input_dict["img"]

        # Fast path: a stacked (num_cams, C, H, W) / (num_cams, H, W, C) array is
        # written in place, skipping the per-camera copy plus the transpose and
        # re-stack that the generic container round-trip would cost.
        if isinstance(images, np.ndarray) and images.ndim == 4:
            if len(camera_names) != images.shape[0]:
                raise ValueError(
                    f"camera_names has {len(camera_names)} entries but {images.shape[0]} images "
                    "are present; per-camera polygons cannot be matched reliably."
                )
            channels_first = is_chw_image(images[0])
            for index, camera_name in enumerate(camera_names):
                view = np.moveaxis(images[index], 0, -1) if channels_first else images[index]
                self._paint_camera_inplace(view, camera_name)
            return input_dict

        image_list, format_info = as_hwc_image_list(images)
        if len(camera_names) != len(image_list):
            raise ValueError(
                f"camera_names has {len(camera_names)} entries but {len(image_list)} images are "
                "present; per-camera polygons cannot be matched reliably."
            )
        painted = []
        for index, image in enumerate(image_list):
            if self.camera_polygons.get(camera_names[index]) is None:
                painted.append(image)
                continue
            image = image.copy()
            self._paint_camera_inplace(image, camera_names[index])
            painted.append(image)
        input_dict["img"] = restore_image_container(images, painted, format_info)
        return input_dict

    def _masked_indices(
        self, camera_name: str, height: int, width: int
    ) -> tuple[npt.NDArray, npt.NDArray] | None:
        """Return cached row/column indices of one camera's masked pixels."""
        polygon = self.camera_polygons.get(camera_name)
        if polygon is None:
            return None

        key = (camera_name, height, width)
        cached = self._index_cache.get(key)
        if cached is None:
            mask = np.zeros((height, width), dtype=np.uint8)
            cv2.fillPoly(mask, [mask_polygon_to_pixels(polygon, width, height)], 255)
            cached = np.nonzero(mask)
            self._index_cache[key] = cached
        return cached

    def _paint_camera_inplace(self, hwc_image: npt.NDArray, camera_name: str) -> None:
        """Paint one HWC image view in place if its camera has a polygon.

        ``hwc_image`` must be a view onto the caller's buffer (or a copy the
        caller is willing to have modified); the fill is written through it.
        """
        indices = self._masked_indices(camera_name, hwc_image.shape[0], hwc_image.shape[1])
        if indices is None:
            return
        hwc_image[indices] = np.asarray(self.mask_color, dtype=hwc_image.dtype)
