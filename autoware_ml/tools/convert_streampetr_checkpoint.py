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

"""Prepare an image-backbone checkpoint for StreamPETR initialization.

StreamPETR trains end to end inside autoware-ml (nuScenes pretrain -> T4 base ->
j6gen2 fine-tune) and passes Lightning checkpoints straight between stages, so
no conversion happens inside the flow. The single external artifact it still
needs is the DD3D/FCOS3D-pretrained VoVNet-99 image backbone published by the
upstream StreamPETR release, whose ``img_backbone.*`` names already match the
native VoVNet layout.

This script keeps those tensors, drops everything else (the release also carries
FCOS3D's own 2D detection head), and optionally flips the stem convolution's
input channels so BGR-trained weights consume the RGB images this pipeline
loads. Run it once; the result is reusable for every pretrain run.

Usage:
    python -m autoware_ml.tools.convert_streampetr_checkpoint \
        --input fcos3d_vovnet_imgbackbone-remapped.pth \
        --output fcos3d_vovnet_imgbackbone_rgb_native.pth \
        --bgr-to-rgb
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch

logger = logging.getLogger(__name__)

_BACKBONE_PREFIX = "img_backbone."
_STEM_CONV_KEY = "img_backbone.stem.stem_1/conv.weight"


def convert_state_dict(
    state_dict: dict[str, torch.Tensor],
    *,
    bgr_to_rgb: bool,
) -> tuple[dict[str, torch.Tensor], list[str]]:
    """Keep the image-backbone tensors, optionally flipping the stem to RGB.

    Args:
        state_dict: Source state dict. Keys outside ``img_backbone.`` are dropped.
        bgr_to_rgb: Flip the stem convolution's input channels so BGR-trained
            weights consume RGB inputs. Valid because the normalization
            statistics swap channel-consistently; the resulting features agree
            to ~1e-6 relative error rather than bitwise, because reordering the
            channels reorders a floating-point sum.

    Returns:
        The backbone state dict and the sorted list of dropped source keys.
    """
    converted = {
        name: tensor for name, tensor in state_dict.items() if name.startswith(_BACKBONE_PREFIX)
    }
    dropped = sorted(name for name in state_dict if not name.startswith(_BACKBONE_PREFIX))

    if bgr_to_rgb:
        if _STEM_CONV_KEY not in converted:
            raise KeyError(
                f"--bgr-to-rgb requested but {_STEM_CONV_KEY!r} is missing from the checkpoint."
            )
        converted[_STEM_CONV_KEY] = converted[_STEM_CONV_KEY][:, [2, 1, 0]].contiguous()

    return converted, dropped


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    """Load a plain state dict, unwrapping a ``{"state_dict": ...}`` payload.

    ``weights_only=True`` is deliberate: this tool only ever reads tensors, so a
    checkpoint carrying pickled framework objects fails loudly instead of
    executing them during load.
    """
    payload = torch.load(str(path), map_location="cpu", weights_only=True)
    return payload.get("state_dict", payload)


def main() -> None:
    """Convert one image-backbone checkpoint."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True, help="source checkpoint (.pth)")
    parser.add_argument("--output", type=Path, required=True, help="converted checkpoint path")
    parser.add_argument(
        "--bgr-to-rgb",
        action="store_true",
        help=(
            "Flip the stem conv input channels. Use when the source was trained "
            "on BGR images (the mmcv convention) and the target pipeline loads "
            "RGB, as autoware-ml does."
        ),
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    converted, dropped = convert_state_dict(load_state_dict(args.input), bgr_to_rgb=args.bgr_to_rgb)

    logger.info("Kept %d backbone tensors; dropped %d.", len(converted), len(dropped))
    for name in dropped:
        logger.info("  dropped: %s", name)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": converted}, str(args.output))
    logger.info("Wrote %s", args.output)


if __name__ == "__main__":
    main()
