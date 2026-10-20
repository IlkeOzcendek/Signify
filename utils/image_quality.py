"""Check upload quality and preview the exact image seen by the model.""" # Docstring = Official description stored for this module

import os
import tempfile
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from preprocess import preprocess_signature


@dataclass
class QualityResult: # dataclass automatically creates the constructor for these stored fields
    processed_image: Image.Image # Final grayscale image sent to the model
    warnings: list[str] # Quality concerns that do not block verification
    errors: list[str] # Problems severe enough to make the upload unusable
    blur_score: float # Sharpness estimate calculated from Laplacian variance
    ink_ratio: float # Dark pixels / All processed pixels

    @property
    def usable(self):
        return not self.errors # property allows result.usable without calling it like a function


def inspect_uploaded_signature(uploaded_file, preprocess_mode = "default"):
    """Inspect an uploaded Streamlit file without modifying the source."""

    suffix = Path(uploaded_file.name).suffix or ".png" # Uses PNG when the upload has no extension
    raw_bytes = uploaded_file.getvalue() # Original file content as bytes
    raw_array = np.frombuffer(raw_bytes, dtype = np.uint8) # Converts bytes into an unsigned 8-bit NumPy array
    raw = cv2.imdecode(raw_array, cv2.IMREAD_COLOR) # Decodes the in-memory array as a BGR image

    if raw is None:
        blank = Image.new("L", (224, 224), color = 255) # White fallback keeps the expected grayscale size
        return QualityResult(blank, [], ["The image could not be read; choose another file."], 0.0, 0.0)

    input_path = None # Temporary path of the raw upload
    output_path = None # Temporary path of the processed image

    try:
        with tempfile.NamedTemporaryFile(delete = False, suffix = suffix) as input_file:
            input_file.write(raw_bytes)
            input_path = input_file.name

        with tempfile.NamedTemporaryFile(delete = False, suffix = ".png") as output_file:
            output_path = output_file.name

        preprocess_signature(input_path, output_path, mode = preprocess_mode) # Uses the exact preprocessing route used before model inference
        processed = cv2.imread(output_path, cv2.IMREAD_GRAYSCALE) # Processed one-channel image
    finally:
        if input_path and os.path.exists(input_path):
            os.remove(input_path)

        if output_path and os.path.exists(output_path):
            os.remove(output_path)

    gray = cv2.cvtColor(raw, cv2.COLOR_BGR2GRAY) # Removes color because sharpness depends on edges not color
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var()) # Laplacian variance = Edge variation ------> lower usually means blurrier
    ink_ratio = float((processed < 128).mean()) # Ink ratio = Pixels darker than 128 / All pixels

    warnings = [] # Non-blocking feedback shown to the user
    errors = [] # Blocking quality failures

    if ink_ratio < 0.003:
        errors.append("No signature was detected or the image is almost empty.")
    elif ink_ratio < 0.008:
        warnings.append("The signature looks very thin or faint; try a clearer image.")

    if ink_ratio > 0.18:
        warnings.append("The image contains too many dark areas; check shadows and the background.")

    if blur_score < 25:
        warnings.append("The image looks blurry; hold the camera steady and capture it again.")

    blurred = cv2.GaussianBlur(gray, (3, 3), 0) # Gaussian blur removes tiny noise before edge-contact detection

    _, raw_ink = cv2.threshold(
        blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU
    ) # Otsu automatically chooses the light-dark boundary and inverse makes ink white

    h, w = raw_ink.shape # h w = Image height and width
    band = max(2, int(min(h, w) * 0.015)) # Outer inspection band = 1.5% of the shorter image side and at least two pixels

    edge_pixels = np.concatenate([
        raw_ink[:band, :].ravel(), raw_ink[-band:, :].ravel(),
        raw_ink[:, :band].ravel(), raw_ink[:, -band:].ravel()
    ]) # Combines top bottom left and right border pixels into one array

    if float((edge_pixels > 0).mean()) > 0.025:
        warnings.append("The signature or frame touches the image edge; keep the complete signature inside the frame.")

    return QualityResult(
        Image.fromarray(processed), warnings, errors, blur_score, ink_ratio
    )
