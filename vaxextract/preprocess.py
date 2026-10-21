"""Image clean-up for phone photos/scans: downscale, illumination/colour-cast normalisation, deskew."""
from __future__ import annotations

import cv2
import numpy as np


def estimate_skew(gray: np.ndarray, max_angle: float = 10.0, step: float = 0.5) -> float:
    """Projection-profile skew estimate in degrees (positive = counter-clockwise text tilt)."""
    h, w = gray.shape[:2]
    s = 800.0 / max(h, w)
    small = cv2.resize(gray, (max(1, int(w * s)), max(1, int(h * s))), interpolation=cv2.INTER_AREA) if s < 1 else gray
    bw = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    if cv2.countNonZero(bw) < 200:
        return 0.0
    sh, sw = bw.shape
    centre = (sw / 2, sh / 2)
    best_angle, best_score = 0.0, -1.0
    for a in np.arange(-max_angle, max_angle + 1e-6, step):
        m = cv2.getRotationMatrix2D(centre, float(a), 1.0)
        rot = cv2.warpAffine(bw, m, (sw, sh), flags=cv2.INTER_NEAREST)
        prof = rot.sum(axis=1).astype(np.float64)
        score = float(np.sum(np.diff(prof) ** 2))
        if score > best_score:
            best_angle, best_score = float(a), score
    return best_angle


def deskew(gray: np.ndarray, min_angle: float = 0.3) -> np.ndarray:
    a = estimate_skew(gray)
    if abs(a) < min_angle:
        return gray
    h, w = gray.shape[:2]
    m = cv2.getRotationMatrix2D((w / 2, h / 2), a, 1.0)
    return cv2.warpAffine(gray, m, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_CONSTANT, borderValue=255)


def normalize_illumination(gray: np.ndarray) -> np.ndarray:
    """Divide by an estimated background: removes colour casts, shadows, glare gradients."""
    bg = cv2.GaussianBlur(cv2.dilate(gray, np.ones((15, 15), np.uint8)), (0, 0), sigmaX=25)
    return cv2.divide(gray, bg, scale=255)


def preprocess(rgb: np.ndarray, max_side: int = 2400) -> np.ndarray:
    """RGB/gray uint8 image -> cleaned grayscale uint8 image."""
    if rgb.ndim == 3 and rgb.shape[2] == 4:
        rgb = cv2.cvtColor(rgb, cv2.COLOR_RGBA2RGB)
    h, w = rgb.shape[:2]
    scale = max_side / max(h, w)
    if scale < 1:
        rgb = cv2.resize(rgb, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY) if rgb.ndim == 3 else rgb
    gray = normalize_illumination(gray)
    return deskew(gray)
