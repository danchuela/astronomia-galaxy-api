from __future__ import annotations

import numpy as np


def create_synthetic_image(size: tuple[int, int] = (128, 128)) -> np.ndarray:
    y = np.linspace(-1.0, 1.0, size[0])
    x = np.linspace(-1.0, 1.0, size[1])
    xx, yy = np.meshgrid(x, y)
    sigma_x, sigma_y = 0.28, 0.18
    blob = np.exp(-((xx**2) / (2 * sigma_x**2) + (yy**2) / (2 * sigma_y**2)))
    return normalize_image(blob).astype(np.float32)


def normalize_image(image: np.ndarray) -> np.ndarray:
    arr = image.astype(np.float64)
    min_val = float(np.min(arr))
    max_val = float(np.max(arr))
    denom = (max_val - min_val) + 1e-9
    return (arr - min_val) / denom
