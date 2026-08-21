"""Private helpers shared by camera transform modules."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import numpy.typing as npt


def is_chw_image(image: npt.NDArray) -> bool:
    """Return whether an image uses channel-first layout."""
    return image.ndim == 3 and image.shape[0] in (1, 3, 4) and image.shape[-1] not in (1, 3, 4)


def as_hwc_image_list(images: Any) -> tuple[list[npt.NDArray], dict[str, Any]]:
    """Normalize image containers to a list of HWC images."""
    if isinstance(images, list):
        format_info = {
            "container": "list",
            "layout": "chw" if images and is_chw_image(images[0]) else "hwc",
        }
        image_list = images
    elif isinstance(images, np.ndarray) and images.ndim == 4:
        format_info = {
            "container": "stack",
            "layout": "chw" if images.shape[1] in (1, 3, 4) else "hwc",
        }
        image_list = [images[index] for index in range(images.shape[0])]
    else:
        format_info = {"container": "single", "layout": "chw" if is_chw_image(images) else "hwc"}
        image_list = [images]

    hwc_images = [
        np.transpose(image, (1, 2, 0)) if format_info["layout"] == "chw" else image
        for image in image_list
    ]
    return hwc_images, format_info


def restore_image_container(
    template: Any, images: list[npt.NDArray], format_info: dict[str, Any]
) -> Any:
    """Restore a list of HWC images to the original container type."""
    restored = [
        np.transpose(image, (2, 0, 1)) if format_info["layout"] == "chw" else image
        for image in images
    ]
    if format_info["container"] == "list":
        return restored
    if format_info["container"] == "stack":
        return np.stack(restored, axis=0)
    del template
    return restored[0]


def normalize_mask_polygon(polygon: Sequence[float]) -> npt.NDArray:
    """Validate and reshape a flat ``[x0, y0, x1, y1, ...]`` polygon.

    Args:
        polygon: Flat sequence of normalized ``[0, 1]`` image-space x/y pairs.

    Returns:
        Polygon vertices with shape ``(K, 2)``.
    """
    points = np.asarray(polygon, dtype=np.float32).reshape(-1, 2)
    if points.shape[0] < 3:
        raise ValueError("Mask polygon requires at least 3 normalized x/y points.")
    if np.any((points < 0.0) | (points > 1.0)):
        raise ValueError("Mask polygon values must be normalized to [0, 1].")
    return points


def mask_polygon_to_pixels(polygon: npt.NDArray, width: int, height: int) -> npt.NDArray:
    """Scale a normalized polygon to integer pixel coordinates.

    Args:
        polygon: Normalized polygon vertices with shape ``(K, 2)``.
        width: Image width in pixels.
        height: Image height in pixels.

    Returns:
        Integer pixel-space polygon vertices with shape ``(K, 2)``.
    """
    pixels = polygon.copy()
    pixels[:, 0] *= width - 1
    pixels[:, 1] *= height - 1
    return np.rint(pixels).astype(np.int32)


def points_inside_polygon(points: npt.NDArray, polygon: npt.NDArray) -> npt.NDArray:
    """Test each point for containment in a polygon via ray casting.

    Args:
        points: Query points with shape ``(N, 2)`` in the same space as ``polygon``.
        polygon: Polygon vertices with shape ``(K, 2)``.

    Returns:
        Boolean array of shape ``(N,)``, ``True`` where the point is inside.
    """
    x, y = points[:, 0], points[:, 1]
    poly_x, poly_y = polygon[:, 0], polygon[:, 1]
    inside = np.zeros(len(points), dtype=bool)
    j = len(polygon) - 1
    for i in range(len(polygon)):
        crosses = ((poly_y[i] > y) != (poly_y[j] > y)) & (
            x <= (poly_x[j] - poly_x[i]) * (y - poly_y[i]) / (poly_y[j] - poly_y[i] + 1e-12) + poly_x[i]
        )
        inside ^= crosses
        j = i
    return inside
