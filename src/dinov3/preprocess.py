"""
Reusable thumbnail preprocessing for DINOv3.

Pipeline:
  1. Filter invalid / placeholder files
  2. Convert to RGB
  3. Letterbox to square (preserve 16:9 composition)
  4. Resize to model input (224 for CLS, 518 for patches)
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Tuple

from PIL import Image

# Matches VLM pipeline and acquisition validity checks in docs/results.md
DEFAULT_MIN_BYTES = 4096
DEFAULT_VALID_SIZE = (640, 360)
LETTERBOX_FILL = (0, 0, 0)


@dataclass(frozen=True)
class PreprocessResult:
    source_path: Path
    source_size: Tuple[int, int]
    source_mode: str
    source_bytes: int
    square_size: int
    target_size: int
    letterbox_pad_top: int
    image: Image.Image


def is_valid_thumbnail(
    path: Path,
    *,
    min_bytes: int = DEFAULT_MIN_BYTES,
    valid_size: Tuple[int, int] = DEFAULT_VALID_SIZE,
) -> Tuple[bool, str]:
    """Return (ok, reason). Aligns with vlm_annotate.py size filter."""
    if not path.exists():
        return False, "missing"
    try:
        nbytes = path.stat().st_size
    except OSError as exc:
        return False, f"stat_error:{exc}"

    if nbytes < min_bytes:
        return False, f"too_small:{nbytes}B"

    try:
        with Image.open(path) as im:
            if im.size != valid_size:
                return False, f"unexpected_size:{im.size[0]}x{im.size[1]}"
    except OSError as exc:
        return False, f"open_error:{exc}"

    return True, "ok"


def load_rgb_image(path: Path) -> Image.Image:
    with Image.open(path) as im:
        return im.convert("RGB")


def letterbox_to_square(image: Image.Image, fill: Tuple[int, int, int] = LETTERBOX_FILL) -> Tuple[Image.Image, int]:
    """Scale to fit inside a square canvas; return (image, top_padding_px)."""
    w, h = image.size
    side = max(w, h)
    if w >= h:
        new_w = side
        new_h = int(round(h * side / w))
    else:
        new_h = side
        new_w = int(round(w * side / h))
    resized = image.resize((new_w, new_h), Image.BICUBIC)

    canvas = Image.new("RGB", (side, side), fill)
    pad_left = (side - new_w) // 2
    pad_top = (side - new_h) // 2
    canvas.paste(resized, (pad_left, pad_top))
    return canvas, pad_top


def resize_square(image: Image.Image, target: int) -> Image.Image:
    if image.size[0] == image.size[1] == target:
        return image
    return image.resize((target, target), Image.BICUBIC)


def preprocess_for_dinov3(
    path: Path,
    *,
    target_size: int = 518,
    min_bytes: int = DEFAULT_MIN_BYTES,
    valid_size: Tuple[int, int] = DEFAULT_VALID_SIZE,
) -> PreprocessResult:
    ok, reason = is_valid_thumbnail(path, min_bytes=min_bytes, valid_size=valid_size)
    if not ok:
        raise ValueError(f"{path.name}: {reason}")

    rgb = load_rgb_image(path)
    squared, pad_top = letterbox_to_square(rgb)
    final = resize_square(squared, target_size)
    return PreprocessResult(
        source_path=path,
        source_size=rgb.size,
        source_mode="RGB",
        source_bytes=path.stat().st_size,
        square_size=squared.size[0],
        target_size=target_size,
        letterbox_pad_top=pad_top,
        image=final,
    )