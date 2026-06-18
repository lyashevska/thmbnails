"""
DINOv3 CLS embedding extraction for preprocessed thumbnails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

from .config import DEFAULT_PATCH_SIZE

import numpy as np
from PIL import Image

from .config import DEFAULT_CLS_SIZE, DEFAULT_MODEL_ID
from .preprocess import PreprocessResult, preprocess_for_dinov3

try:
    import torch
    from transformers import AutoImageProcessor, AutoModel
except ImportError as exc:
    torch = None  # type: ignore
    AutoImageProcessor = None  # type: ignore
    AutoModel = None  # type: ignore
    _IMPORT_ERROR = exc
else:
    _IMPORT_ERROR = None


@dataclass
class Dinov3Bundle:
    model_id: str
    device: str
    processor: "AutoImageProcessor"
    model: "AutoModel"
    cls_size: int


@dataclass
class PatchExtractResult:
    patches: np.ndarray
    rows: np.ndarray
    cols: np.ndarray
    grid_shape: Tuple[int, int]
    patch_size: int
    preprocessed: PreprocessResult


def _require_torch() -> None:
    if _IMPORT_ERROR is not None:
        raise ImportError(
            "DINOv3 extraction requires torch and transformers. "
            "Install with: pip install -r requirements-dinov3.txt"
        ) from _IMPORT_ERROR


def resolve_device(requested: Optional[str] = None) -> str:
    _require_torch()
    if requested:
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_dinov3(
    model_id: str = DEFAULT_MODEL_ID,
    *,
    device: Optional[str] = None,
    cls_size: int = DEFAULT_CLS_SIZE,
) -> Dinov3Bundle:
    _require_torch()
    resolved = resolve_device(device)
    processor = AutoImageProcessor.from_pretrained(model_id)
    model = AutoModel.from_pretrained(model_id)
    model.to(resolved)
    model.eval()
    return Dinov3Bundle(
        model_id=model_id,
        device=resolved,
        processor=processor,
        model=model,
        cls_size=cls_size,
    )


def pil_to_pixel_values(bundle: Dinov3Bundle, image: Image.Image) -> "torch.Tensor":
    """Normalize a letterboxed PIL image without re-resizing or cropping."""
    inputs = bundle.processor(
        images=image,
        return_tensors="pt",
        do_resize=False,
        do_center_crop=False,
    )
    return inputs["pixel_values"].to(bundle.device)


def extract_cls_from_preprocessed(
    bundle: Dinov3Bundle,
    preprocessed: PreprocessResult,
) -> np.ndarray:
    pixel_values = pil_to_pixel_values(bundle, preprocessed.image)
    with torch.inference_mode():
        outputs = bundle.model(pixel_values=pixel_values)

    if outputs.pooler_output is not None:
        vector = outputs.pooler_output.squeeze(0)
    else:
        vector = outputs.last_hidden_state[:, 0, :].squeeze(0)

    return vector.detach().cpu().float().numpy()


def extract_cls_from_path(
    bundle: Dinov3Bundle,
    path: Path,
    *,
    min_bytes: int = 4096,
) -> Tuple[np.ndarray, PreprocessResult]:
    preprocessed = preprocess_for_dinov3(
        path,
        target_size=bundle.cls_size,
        min_bytes=min_bytes,
    )
    embedding = extract_cls_from_preprocessed(bundle, preprocessed)
    return embedding, preprocessed


def content_row_mask(
    grid_h: int,
    *,
    pad_top_px: int,
    content_height_px: int,
    patch_size: int,
) -> np.ndarray:
    """True for patch rows whose center falls inside the letterboxed content band."""
    rows = np.arange(grid_h, dtype=np.int32)
    centers = (rows + 0.5) * patch_size
    return (centers >= pad_top_px) & (centers < pad_top_px + content_height_px)


def letterbox_content_bounds(preprocessed: PreprocessResult) -> Tuple[int, int]:
    """Return (pad_top, content_height) in preprocessed target pixels."""
    square = preprocessed.square_size
    target = preprocessed.target_size
    scale = target / square
    pad_top = int(round(preprocessed.letterbox_pad_top * scale))
    source_h = preprocessed.source_size[1]
    content_h = int(round(source_h * (target / square)))
    return pad_top, content_h


def extract_patches_from_preprocessed(
    bundle: Dinov3Bundle,
    preprocessed: PreprocessResult,
) -> PatchExtractResult:
    pixel_values = pil_to_pixel_values(bundle, preprocessed.image)
    patch_size = int(bundle.model.config.patch_size)
    _, _, height, width = pixel_values.shape
    grid_h, grid_w = height // patch_size, width // patch_size

    with torch.inference_mode():
        outputs = bundle.model(pixel_values=pixel_values)

    hidden = outputs.last_hidden_state.squeeze(0)
    num_register = int(getattr(bundle.model.config, "num_register_tokens", 0) or 0)
    patch_tokens = hidden[1 + num_register :, :].detach().cpu().float().numpy()
    expected = grid_h * grid_w
    if patch_tokens.shape[0] != expected:
        raise ValueError(f"Expected {expected} patch tokens, got {patch_tokens.shape[0]}")

    patch_grid = patch_tokens.reshape(grid_h, grid_w, -1)
    pad_top, content_h = letterbox_content_bounds(preprocessed)
    row_mask = content_row_mask(
        grid_h,
        pad_top_px=pad_top,
        content_height_px=content_h,
        patch_size=patch_size,
    )

    valid_rows = np.where(row_mask)[0]
    if len(valid_rows) == 0:
        raise ValueError("No content patches after letterbox mask")
    rows = np.repeat(valid_rows, grid_w)
    cols = np.tile(np.arange(grid_w, dtype=np.int32), len(valid_rows))
    patches = patch_grid[rows, cols, :]
    return PatchExtractResult(
        patches=patches,
        rows=rows.astype(np.int32),
        cols=cols.astype(np.int32),
        grid_shape=(grid_h, grid_w),
        patch_size=patch_size,
        preprocessed=preprocessed,
    )


def extract_patches_from_path(
    bundle: Dinov3Bundle,
    path: Path,
    *,
    patch_size: int = DEFAULT_PATCH_SIZE,
    min_bytes: int = 4096,
) -> PatchExtractResult:
    preprocessed = preprocess_for_dinov3(
        path,
        target_size=patch_size,
        min_bytes=min_bytes,
    )
    return extract_patches_from_preprocessed(bundle, preprocessed)