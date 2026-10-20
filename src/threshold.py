"""Provide one shared source for every signature decision threshold.

evaluate.py calculates and saves the values while test.py verify.py and the application read them.

threshold        ------> Accuracy-optimal binary decision boundary
t_confident_same ------> Lower distances represent a confident match
t_confident_diff ------> Higher distances represent a confident mismatch
The area between both confidence boundaries represents an uncertain decision.
"""

import json
from pathlib import Path

THRESHOLD_PATH = Path("best_threshold.json") # Shared JSON file containing the saved decision boundaries


def save_thresholds(threshold, t_confident_same, t_confident_diff):
    THRESHOLD_PATH.write_text(json.dumps({
        "threshold": float(threshold), # Converts NumPy values into numbers JSON can serialize
        "t_confident_same": float(t_confident_same),
        "t_confident_diff": float(t_confident_diff)
    }, indent = 2)) # indent creates a readable two-space JSON layout


def save_threshold(value):
    THRESHOLD_PATH.write_text(json.dumps({"threshold": float(value)}, indent = 2)) # Backward-compatible writer for one binary threshold


def _load():
    if THRESHOLD_PATH.exists():
        return json.loads(THRESHOLD_PATH.read_text()) # Converts saved JSON text back into a Python dictionary

    return None


def load_threshold(default = 1.0):
    data = _load() # data = Saved threshold dictionary or None when the file is missing

    if data is not None:
        return float(data["threshold"])

    print(
        f"WARNING: {THRESHOLD_PATH} was not found; default threshold {default} "
        f"is being used. Run 'python evaluate.py' first to calculate the correct threshold."
    )

    return default


def load_bands(default_same = 0.4, default_diff = 0.8):
    data = _load()

    if data is not None and "t_confident_same" in data:
        return float(data["t_confident_same"]), float(data["t_confident_diff"]) # Returns match and mismatch confidence boundaries

    return default_same, default_diff
