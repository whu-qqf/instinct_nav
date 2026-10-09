from __future__ import annotations

from pathlib import Path
from typing import Protocol

import cv2
import numpy as np


class DynamicMaskProvider(Protocol):
    """Produce a binary dynamic-object mask in the input RGB pixel frame."""

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        ...


class Yolo11Segmentation:
    """Ultralytics YOLO11 instance segmentation provider."""

    def __init__(
        self,
        weights: str | Path,
        *,
        class_ids: tuple[int, ...] | None = None,
        confidence: float = 0.25,
        iou: float = 0.45,
        image_size: int = 640,
        device: str = "0",
        half: bool = True,
        max_detections: int = 20,
    ) -> None:
        model_path = _require_file(weights, "YOLO11-seg weights")

        try:
            from ultralytics import YOLO
        except ImportError as exc:
            raise ModuleNotFoundError(
                "YOLO11-seg dependencies are incomplete. Install deploy/requirements-yolo.txt."
            ) from exc

        self.model = YOLO(str(model_path), task="segment")
        self.device = str(device)
        self.class_ids = list(class_ids) if class_ids is not None else None
        self.confidence = float(confidence)
        self.iou = float(iou)
        self.image_size = max(int(image_size), 32)
        self.half = bool(half and self.device.lower() != "cpu")
        self.max_detections = max(int(max_detections), 1)

    def predict(self, rgb: np.ndarray) -> np.ndarray:
        image = _validate_rgb(rgb)
        results = self.model.predict(
            source=image,
            conf=self.confidence,
            iou=self.iou,
            imgsz=self.image_size,
            device=self.device,
            quantize=16 if self.half else None,
            classes=self.class_ids,
            max_det=self.max_detections,
            retina_masks=True,
            verbose=False,
        )
        if not results or results[0].masks is None:
            return np.zeros(image.shape[:2], dtype=np.float32)
        masks = results[0].masks.data
        if masks.shape[0] == 0:
            return np.zeros(image.shape[:2], dtype=np.float32)
        union = masks.bool().any(dim=0).to(dtype=masks.dtype).cpu().numpy()
        return resize_binary_mask(union, image.shape[:2])


def resize_binary_mask(mask: np.ndarray, shape: tuple[int, int]) -> np.ndarray:
    """Resize a binary mask to ``(height, width)`` without softening labels."""

    mask_array = np.asarray(mask)
    if mask_array.ndim == 3 and mask_array.shape[0] == 1:
        mask_array = mask_array[0]
    if mask_array.ndim != 2:
        raise ValueError(f"dynamic mask must be 2D, got shape {mask_array.shape}")
    if mask_array.shape != shape:
        mask_array = cv2.resize(
            mask_array.astype(np.float32), (shape[1], shape[0]), interpolation=cv2.INTER_NEAREST
        )
    return (mask_array >= 0.5).astype(np.float32)


def _validate_rgb(rgb: np.ndarray) -> np.ndarray:
    image = np.asarray(rgb)
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError(f"model input must be HxWx3 RGB, got shape {image.shape}")
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    return np.ascontiguousarray(image)


def _require_file(path: str | Path, label: str) -> Path:
    resolved = Path(path).expanduser().resolve()
    if not resolved.is_file():
        raise FileNotFoundError(f"{label} not found: {resolved}")
    return resolved
