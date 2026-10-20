"""Small rear-camera-first Streamlit component for mobile signature capture."""

import base64
from pathlib import Path

import streamlit.components.v1 as components

from utils.camera_capture import PreparedImage


_FRONTEND_DIR = Path(__file__).resolve().parents[1] / "assets" / "mobile_camera"
_camera_component = components.declare_component(
    "signify_mobile_camera",
    path=str(_FRONTEND_DIR),
)


def mobile_camera(key: str):
    """Return the latest JPEG capture or None before the shutter is pressed."""

    value = _camera_component(key=key, default=None)
    if not value or not isinstance(value, str) or "," not in value:
        return None

    header, encoded = value.split(",", 1)
    if not header.startswith("data:image/"):
        return None

    try:
        data = base64.b64decode(encoded, validate=True)
    except (ValueError, TypeError):
        return None

    return PreparedImage(data, "camera_signature.jpg")
