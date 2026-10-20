"""Experimental grayscale-preserving preprocessing for low-resolution scans."""

import hashlib
from pathlib import Path

import cv2
import numpy as np
import torch


TARGET_SIZE = 224
INNER_SIZE = 190
CACHE_DIR = Path(".cache/archive_grayscale_v2")


def preprocess_array(path):
    image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    image = cv2.GaussianBlur(image, (3, 3), 0)

    # Use a binary mask only to locate ink; retain original gray intensities.
    _, mask = cv2.threshold(
        image, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    )
    h, w = mask.shape
    band_y = max(1, int(h * 0.12))
    band_x = max(1, int(w * 0.12))
    horizontal = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (max(int(w * 0.55), 1), 1)),
    )
    vertical = cv2.morphologyEx(
        mask, cv2.MORPH_OPEN,
        cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(int(h * 0.55), 1))),
    )
    edge_mask = np.zeros_like(mask)
    edge_mask[:band_y] = 255; edge_mask[-band_y:] = 255
    edge_mask[:, :band_x] = 255; edge_mask[:, -band_x:] = 255
    remove = cv2.bitwise_and(cv2.bitwise_or(horizontal, vertical), edge_mask)
    mask = cv2.subtract(mask, remove)
    points = cv2.findNonZero(mask)
    if points is None:
        return np.full((TARGET_SIZE, TARGET_SIZE), 255, np.uint8)

    x, y, cw, ch = cv2.boundingRect(points)
    padding = int(max(cw, ch) * 0.08)
    x1, y1 = max(0, x - padding), max(0, y - padding)
    x2, y2 = min(w, x + cw + padding), min(h, y + ch + padding)
    crop_gray = image[y1:y2, x1:x2]
    crop_mask = mask[y1:y2, x1:x2]
    # Whiten detected border lines while keeping antialiased stroke intensity.
    crop_gray = crop_gray.copy()
    crop_gray[crop_mask == 0] = 255
    ch, cw = crop_gray.shape
    scale = min(INNER_SIZE / max(cw, 1), INNER_SIZE / max(ch, 1))
    nw, nh = max(1, round(cw * scale)), max(1, round(ch * scale))
    resized = cv2.resize(crop_gray, (nw, nh), interpolation=cv2.INTER_AREA)
    canvas = np.full((TARGET_SIZE, TARGET_SIZE), 255, np.uint8)
    ox, oy = (TARGET_SIZE - nw) // 2, (TARGET_SIZE - nh) // 2
    canvas[oy:oy + nh, ox:ox + nw] = resized
    return canvas


def load_image(path):
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    key = hashlib.md5(str(Path(path).resolve()).encode()).hexdigest()
    cache = CACHE_DIR / f"{key}.png"
    if not cache.exists():
        cv2.imwrite(str(cache), preprocess_array(path))
    image = cv2.imread(str(cache), cv2.IMREAD_GRAYSCALE)
    return torch.from_numpy(image).float().unsqueeze(0) / 255.0
