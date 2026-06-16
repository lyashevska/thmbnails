"""
DINOv3 CLS embedding extraction for preprocessed thumbnails.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Tuple

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