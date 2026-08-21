# Copyright 2026 TIER IV, Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""3D bounding-box filter transforms."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from autoware_ml.transforms.base import BaseTransform
from autoware_ml.transforms.camera.annotations2d import _boxes3d_corners, _project_points
from autoware_ml.transforms.camera.utils import normalize_mask_polygon, points_inside_polygon

_BOX_KEYS = ("gt_boxes", "gt_names", "gt_labels", "gt_num_points")


def _filter_present_box_keys(input_dict: dict[str, Any], mask: np.ndarray) -> None:
    """Apply one per-box mask to every present box-aligned annotation key."""
    for key in _BOX_KEYS:
        if key in input_dict:
            input_dict[key] = input_dict[key][mask]


def _resolve_point_coords(input_dict: dict[str, Any]) -> np.ndarray:
    """Return ``(N, 3)`` point coordinates from ``coord`` or ``points``.

    PTv3-style pipelines split the cloud into ``coord``; pillar pipelines keep
    the raw ``points`` array (consumed downstream by the voxel preprocessor).
    Either is acceptable for counting points inside boxes.
    """
    for key in ("coord", "points"):
        if key in input_dict:
            return np.asarray(input_dict[key], dtype=np.float32)[:, :3]
    raise KeyError("Point-count filters require a point cloud under 'coord' or 'points'.")


def _count_points_in_rotated_boxes(
    coord: np.ndarray,
    boxes: np.ndarray,
) -> np.ndarray:
    """Count the number of points inside each oriented 3D bounding box.

    Args:
        coord: Point coordinates of shape ``(N, 3)``.
        boxes: Bounding boxes of shape ``(M, 7)`` with columns
            ``[cx, cy, cz, dx, dy, dz, yaw]``.

    Returns:
        Integer array of shape ``(M,)`` with the point count per box.
    """
    counts = np.zeros(len(boxes), dtype=np.int64)
    for i, box in enumerate(boxes):
        cx, cy, cz, dx, dy, dz, yaw = box[:7]
        cos_yaw = np.cos(-yaw)
        sin_yaw = np.sin(-yaw)
        # Translate to box center
        delta = coord[:, :3] - np.array([cx, cy, cz], dtype=np.float32)
        # Rotate into box-local frame (around z-axis)
        local_x = delta[:, 0] * cos_yaw - delta[:, 1] * sin_yaw
        local_y = delta[:, 0] * sin_yaw + delta[:, 1] * cos_yaw
        local_z = delta[:, 2]
        inside = (
            (np.abs(local_x) <= dx / 2.0)
            & (np.abs(local_y) <= dy / 2.0)
            & (np.abs(local_z) <= dz / 2.0)
        )
        counts[i] = inside.sum()
    return counts


class ObjectNameFilter(BaseTransform):
    """Keep only 3D boxes whose class name is in the allowed list.

    Required keys:
        gt_names: Per-box class name array.

    Optional keys:
        gt_boxes: 3D bounding boxes. Filtered when present.
        gt_labels: Per-box label indices. Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_names: Filtered class names.
        gt_boxes: Filtered boxes (when present).
        gt_labels: Filtered labels (when present).
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = ["gt_names"]

    def __init__(self, *, classes: Sequence[str]) -> None:
        """Initialize the ObjectNameFilter transform.

        Args:
            classes: Allowed class names retained in the sample.
        """
        self.classes = set(classes)

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter present box-aligned arrays by allowed class names.

        Args:
            input_dict: Sample dictionary containing ``gt_names``.

        Returns:
            Updated sample dictionary with disallowed classes removed.
        """
        mask = np.array([n in self.classes for n in input_dict["gt_names"]], dtype=bool)
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class ObjectRangeFilter(BaseTransform):
    """Filter 3D bounding boxes and associated labels by point-cloud range.

    Required keys:
        (none)

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes (when present).
        gt_names: Filtered class names (when present alongside gt_boxes).
        gt_labels: Filtered labels (when present alongside gt_boxes).
        gt_num_points: Filtered lidar point counts (when present alongside gt_boxes).
    """

    _required_keys: list[str] = []
    _optional_keys = ["gt_boxes"]

    def __init__(self, *, point_cloud_range: Sequence[float]) -> None:
        """Initialize the ObjectRangeFilter transform.

        Args:
            point_cloud_range: ``[x_min, y_min, z_min, x_max, y_max, z_max]``.
        """
        self.point_cloud_range = np.asarray(point_cloud_range, dtype=np.float32)

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """No defaults needed - transform is a no-op when gt_boxes is absent."""
        pass

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter boxes whose centers fall outside the configured range.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if "gt_boxes" not in input_dict:
            return input_dict

        boxes = input_dict["gt_boxes"]
        pcr = self.point_cloud_range
        mask = (
            (boxes[:, 0] >= pcr[0])
            & (boxes[:, 1] >= pcr[1])
            & (boxes[:, 2] >= pcr[2])
            & (boxes[:, 0] <= pcr[3])
            & (boxes[:, 1] <= pcr[4])
            & (boxes[:, 2] <= pcr[5])
        )
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class ObjectMinPointsFilter(BaseTransform):
    """Remove 3D boxes that contain fewer than a minimum number of points.

    Required keys:
        gt_names: Class name per box.

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Filtered when present.
        coord: Point coordinates (Nx3 or wider). Required when gt_boxes is present.
        points: Raw point array (Nx3 or wider). Required when gt_boxes is present and
            coord is absent.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes (when present).
        gt_names: Filtered class names.
        gt_labels: Filtered labels (when present).
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = ["gt_names"]
    _optional_keys = ["gt_boxes", "coord", "points"]

    def __init__(self, *, min_num_points: int) -> None:
        """Initialize the ObjectMinPointsFilter transform.

        Args:
            min_num_points: Minimum number of points required inside each box.
        """
        self.min_num_points = min_num_points

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """No defaults needed - transform is a no-op when gt_boxes is absent."""
        pass

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove boxes with too few interior points.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary.
        """
        if "gt_boxes" not in input_dict:
            return input_dict

        coord = _resolve_point_coords(input_dict)
        boxes = input_dict["gt_boxes"]
        counts = _count_points_in_rotated_boxes(coord, boxes)
        mask = counts >= self.min_num_points
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class ObjectRangeMinPointsFilter(BaseTransform):
    """Remove boxes below a point-count threshold within a BEV radial interval.

    Required keys:
        gt_names: Class name per box.

    Optional keys:
        gt_boxes: 3D bounding boxes (Nx7 or Nx9). Filtered when present.
        coord: Point coordinates (Nx3 or wider). Required when gt_boxes is present.
        points: Raw point array (Nx3 or wider). Required when gt_boxes is present and
            coord is absent.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes (when present).
        gt_names: Filtered class names.
        gt_labels: Filtered labels (when present).
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = ["gt_names"]
    _optional_keys = ["gt_boxes", "coord", "points"]

    def __init__(self, *, range_radius: Sequence[float], min_num_points: int) -> None:
        """Initialize the ObjectRangeMinPointsFilter transform.

        Args:
            range_radius: Radial interval ``[min_radius, max_radius]`` in meters.
            min_num_points: Minimum points required for boxes inside the interval.
        """
        if len(range_radius) != 2:
            raise ValueError(f"range_radius must contain [min, max], got {range_radius}")
        min_radius, max_radius = (float(value) for value in range_radius)
        if min_radius < 0.0 or min_radius >= max_radius:
            raise ValueError(f"Expected 0 <= min radius < max radius, got {range_radius}")
        if min_num_points <= 0:
            raise ValueError(f"min_num_points must be positive, got {min_num_points}")
        self.min_radius = min_radius
        self.max_radius = max_radius
        self.min_num_points = min_num_points

    def apply_defaults(self, input_dict: dict[str, Any]) -> None:
        """No defaults needed because missing boxes make this transform a no-op."""
        pass

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Filter boxes in the configured radial band by point count.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary with low-support in-range boxes removed.
        """
        if "gt_boxes" not in input_dict:
            return input_dict

        boxes = input_dict["gt_boxes"]
        radii = np.linalg.norm(boxes[:, :2], axis=1)
        in_range = (radii >= self.min_radius) & (radii < self.max_radius)
        counts = _count_points_in_rotated_boxes(_resolve_point_coords(input_dict), boxes)
        mask = ~in_range | (counts >= self.min_num_points)
        _filter_present_box_keys(input_dict, mask)
        return input_dict


class EgoAreaMaskFilter(BaseTransform):
    """Drop 3D boxes whose projection falls entirely inside an ego-body mask.

    Cameras mounted on the ego vehicle can have part of their field of view
    occluded by the vehicle's own body or mirrors. For each camera, a fixed
    normalized polygon marks that occluded region in image space. A box is
    removed only if, in every camera where it projects into the image, its
    clipped 2D bounding box lies entirely inside that camera's polygon - i.e.
    the box is not genuinely visible anywhere else either.

    Required keys:
        gt_boxes: 3D bounding boxes ``(N, >=7)`` as gravity-center
            ``[x, y, z, dx, dy, dz, yaw, ...]``, in lidar frame.
        gt_labels: Per-box label indices, only used to keep filtered arrays aligned.
        lidar2cam: Per-camera lidar-to-camera transforms, shape ``(num_cams, 4, 4)``.
        camera_intrinsics: Per-camera intrinsics, shape ``(num_cams, 3, 3)`` or ``(num_cams, 4, 4)``.
        img: Per-camera images, shape ``(num_cams, C, H, W)`` - only used for image size.
        camera_names: Per-sample camera names in the same order as the stacked
            per-camera arrays. Read from the sample rather than from config
            because ``LoadMultiViewImagesFromFiles(shuffle_order=True)``
            permutes the camera order per sample; indexing a static config
            list would apply each polygon to the wrong camera.

    Optional keys:
        gt_names: Per-box class name array. Filtered when present.
        gt_num_points: Per-box lidar point counts. Filtered when present.

    Generated keys:
        gt_boxes: Filtered boxes.
        gt_names: Filtered class names (when present).
        gt_labels: Filtered labels.
        gt_num_points: Filtered lidar point counts (when present).
    """

    _required_keys = [
        "gt_boxes",
        "gt_labels",
        "lidar2cam",
        "camera_intrinsics",
        "img",
        "camera_names",
    ]

    def __init__(self, *, camera_masks: dict[str, Sequence[float]]) -> None:
        """Initialize the EgoAreaMaskFilter transform.

        Args:
            camera_masks: Mapping from camera name to a flat normalized
                ``[x0, y0, x1, y1, ...]`` polygon marking that camera's
                ego-occluded region. Cameras absent from this mapping are not masked.
        """
        self.camera_polygons = {
            str(camera): normalize_mask_polygon(polygon) for camera, polygon in camera_masks.items()
        }

    def transform(self, input_dict: dict[str, Any]) -> dict[str, Any]:
        """Remove boxes that are ego-masked in every camera they project into.

        Args:
            input_dict: Sample dictionary updated in place.

        Returns:
            Updated sample dictionary with fully ego-masked boxes removed.
        """
        gt_boxes = np.asarray(input_dict["gt_boxes"], dtype=np.float32)
        if gt_boxes.shape[0] == 0:
            return input_dict

        images = input_dict["img"]
        image_height, image_width = int(images.shape[-2]), int(images.shape[-1])
        corners = _boxes3d_corners(gt_boxes)

        camera_names = list(input_dict["camera_names"])
        num_cams = len(input_dict["lidar2cam"])
        if len(camera_names) != num_cams:
            raise ValueError(
                f"camera_names has {len(camera_names)} entries but {num_cams} cameras are stacked; "
                "per-camera polygons cannot be matched reliably."
            )
        # visible_and_unmasked[b, c]: box b projects into camera c and is not
        # fully covered by that camera's ego polygon there.
        visible_and_unmasked = np.zeros((gt_boxes.shape[0], num_cams), dtype=bool)
        projects_anywhere = np.zeros(gt_boxes.shape[0], dtype=bool)

        # Homogeneous corners once for every box: (N * 8, 4). Projecting all
        # boxes of a camera in one matmul is what keeps this off a per-box
        # Python loop; the only remaining loop is over the handful of cameras.
        num_boxes = gt_boxes.shape[0]
        corners_hom = np.concatenate(
            [corners.reshape(-1, 3), np.ones((num_boxes * 8, 1), dtype=corners.dtype)], axis=1
        ).astype(np.float64)

        for camera_index in range(num_cams):
            polygon = self.camera_polygons.get(camera_names[camera_index])
            lidar2cam = np.asarray(input_dict["lidar2cam"][camera_index], dtype=np.float64)
            cam2img = np.asarray(input_dict["camera_intrinsics"][camera_index], dtype=np.float64)

            points_cam = corners_hom @ lidar2cam.T
            in_front = (points_cam[:, 2] > 0).reshape(num_boxes, 8)
            projected = points_cam[:, :3] @ cam2img[:3, :3].T
            pixels = projected[:, :2] / np.maximum(projected[:, 2:3], 1e-6)
            pixels = pixels.reshape(num_boxes, 8, 2)

            # Reduce over corners while ignoring the ones behind the camera, by
            # pushing masked-out corners to +/-inf so they never win min/max.
            has_corner = in_front.any(axis=1)
            xs = np.where(in_front, pixels[..., 0], np.inf)
            ys = np.where(in_front, pixels[..., 1], np.inf)
            x_min = np.clip(xs.min(axis=1), 0, image_width)
            y_min = np.clip(ys.min(axis=1), 0, image_height)
            xs = np.where(in_front, pixels[..., 0], -np.inf)
            ys = np.where(in_front, pixels[..., 1], -np.inf)
            x_max = np.clip(xs.max(axis=1), 0, image_width)
            y_max = np.clip(ys.max(axis=1), 0, image_height)

            projects = has_corner & (x_min != x_max) & (y_min != y_max)
            if not projects.any():
                continue
            projects_anywhere |= projects

            if polygon is None:
                visible_and_unmasked[:, camera_index] |= projects
                continue

            # A rectangle lies inside a convex-enough polygon iff all 4 of its
            # corners do, which is the same test AWML applies per box.
            pixel_polygon = polygon * np.array(
                [image_width - 1, image_height - 1], dtype=np.float32
            )
            box_corners = np.stack(
                [
                    np.stack([x_min, y_min], axis=1),
                    np.stack([x_max, y_min], axis=1),
                    np.stack([x_max, y_max], axis=1),
                    np.stack([x_min, y_max], axis=1),
                ],
                axis=1,
            )
            inside = points_inside_polygon(
                box_corners.reshape(-1, 2).astype(np.float32), pixel_polygon
            ).reshape(num_boxes, 4)
            fully_masked = inside.all(axis=1)
            visible_and_unmasked[:, camera_index] |= projects & ~fully_masked

        # Keep boxes that never project into any camera untouched (some other
        # filter's job) and boxes that are unmasked in at least one camera
        # they do project into. Drop only boxes that project somewhere and
        # are ego-masked in every camera they reach.
        ego_masked_everywhere = projects_anywhere & ~visible_and_unmasked.any(axis=1)
        mask = ~ego_masked_everywhere
        _filter_present_box_keys(input_dict, mask)
        return input_dict
