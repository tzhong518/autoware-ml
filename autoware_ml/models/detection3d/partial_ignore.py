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

"""Partial-ignore support for partially annotated classes.

Some dataset scenes are annotated for every class except a subset (for
T4Dataset: ``traffic_cone`` and ``barrier``). Training on such frames must not
punish background predictions of the un-annotated classes as false positives.
The per-frame ``traffic_cone_barrier_status`` flag marks whether the frame's
scene carries those annotations; when it is ``False``, classification weights
for the ignored class columns are zeroed on negative (background) queries.
"""

from __future__ import annotations

from collections.abc import Sequence

import torch


def resolve_partial_ignore_labels(
    class_names: Sequence[str] | None,
    partial_ignore_classes: Sequence[str] | None,
) -> list[int] | None:
    """Map partially annotated class names to label indices.

    Args:
        class_names: Ordered detector class names.
        partial_ignore_classes: Class names that are only partially annotated.

    Returns:
        Label indices of the partially annotated classes, or ``None`` when
        partial-ignore is disabled.
    """
    if not partial_ignore_classes:
        return None
    if class_names is None:
        raise ValueError("class_names is required when partial_ignore_classes is set.")
    name_to_index = {name: index for index, name in enumerate(class_names)}
    missing = [name for name in partial_ignore_classes if name not in name_to_index]
    if missing:
        raise ValueError(f"partial_ignore_classes {missing} not found in class_names.")
    return [name_to_index[name] for name in partial_ignore_classes]


def normalize_status_flags(
    value: Sequence[bool] | torch.Tensor | None, batch_size: int
) -> list[bool]:
    """Normalize per-sample annotation-status flags to a plain bool list.

    Args:
        value: Batch status value: a stacked tensor, a per-sample sequence of
            bools or scalar tensors (``"list"`` collation), or ``None``.
        batch_size: Expected number of samples. Missing entries default to
            ``True`` (fully annotated).

    Returns:
        One bool per sample.
    """
    if value is None:
        return [True] * batch_size
    if torch.is_tensor(value):
        value = value.detach().cpu().flatten().tolist()
    flags = [bool(item) for item in value]
    flags.extend([True] * (batch_size - len(flags)))
    return flags[:batch_size]
