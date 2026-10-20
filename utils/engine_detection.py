import math

import cv2


def is_low_resolution_scan(path):
    image = cv2.imread(str(path)) # OpenCV loads color images in BGR channel order

    if image is None:
        return False

    height, width = image.shape[:2] # First two dimensions represent image height and width
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean() # Saturation = Average color intensity in the HSV color space

    # Archive profile ------> compact dimensions with some colored ink | grayscale datasets have saturation near zero
    return (
        min(height, width) <= 220
        and max(height, width) <= 600
        and saturation > 1.5
    )


def is_high_resolution_mono_scan(path):
    image = cv2.imread(str(path))

    if image is None:
        return False

    height, width = image.shape[:2]
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean()

    return height >= 450 and width >= 650 and saturation <= 1.5


def is_compact_scan(path):
    """Detect a compact scan or screenshot including monochrome signatures.""" # Docstring = Official description stored for this function

    image = cv2.imread(str(path))

    if image is None:
        return False

    height, width = image.shape[:2]

    return min(height, width) <= 220 and max(height, width) <= 600


def resolve_engine_details(requested_engine, paths):
    if requested_engine != "auto":
        return requested_engine, "Admin/developer override: engine was selected manually." # Manual selection skips automatic detection

    archive_flags = [is_low_resolution_scan(path) for path in paths] # One True or False archive vote for every image
    utsig_flags = [is_high_resolution_mono_scan(path) for path in paths] # One UTSig profile vote for every image
    required = max(1, math.ceil(len(paths) * 0.60)) # Required votes = At least 60% rounded upward and never less than one

    if sum(archive_flags) >= required:
        return "archive", "Low-resolution, colored scan profile detected."

    # A compact manual reference must not be outvoted by one large questioned image ------> archive v2 handles this resolution mismatch
    if len(paths) == 2 and is_compact_scan(paths[0]):
        return "archive", (
            "Compact reference scan detected; low-resolution scan v2 is used "
            "even when the questioned image is larger."
        )

    if sum(utsig_flags) >= required:
        return "utsig", "High-resolution monochrome scan profile detected."

    return "general", "General photo/scan profile detected."


def resolve_engine(requested_engine, paths):
    return resolve_engine_details(requested_engine, paths)[0] # Index 0 returns only the engine name without its explanation
