import re

from src.threshold import load_threshold, load_bands

THRESHOLD = load_threshold() # Binary decision boundary saved by evaluate.py from validation data
T_SAME, T_DIFF = load_bands() # Distance < T_SAME means match | Distance > T_DIFF means no match


def clean_folder_name(name):
    name = name.strip() # Removes whitespace from the beginning and end
    name = re.sub(r"[^\w\s-]", "", name) # Removes characters unsafe or unnecessary in a folder name
    name = re.sub(r"\s+", "_", name) # Replaces one or more spaces with one underscore

    return name


def classify_verdict(distance):
    """Classify a distance as match uncertain or no match.""" # Distance = Difference between both signature embeddings

    if T_SAME < T_DIFF:
        if distance < T_SAME:
            return "match"

        if distance > T_DIFF:
            return "no_match"

        return "uncertain"

    return "match" if distance < THRESHOLD else "no_match" # Overlapping bands fall back to one binary boundary


def calculate_confidence(distance):
    """Map distance to 70-100 confidence in certain regions and 50-65 inside the uncertain band."""

    if T_SAME < T_DIFF:
        if distance < T_SAME:
            frac = distance / max(T_SAME, 1e-6) # frac = Position inside the match region ------> zero is the closest possible match
            return round(100 - frac * 30, 1) # Maps the match region from 100% down to 70%

        if distance > T_DIFF:
            span = max(2.0 - T_DIFF, 1e-6) # span = Available normalized distance above the no-match boundary
            frac = min((distance - T_DIFF) / span, 1)
            return round(70 + frac * 30, 1) # Maps the no-match region from 70% up to 100%

        mid = (T_SAME + T_DIFF) / 2 # mid = Center of the uncertain band
        half = max((T_DIFF - T_SAME) / 2, 1e-6) # 1e-6 prevents division by zero

        return round(50 + (abs(distance - mid) / half) * 15, 1) # Maps uncertainty from 50% at the center to 65% near either boundary

    difference = abs(distance - THRESHOLD) # Distance from the binary decision boundary when confidence bands overlap

    return round(min(100, 50 + (difference / max(THRESHOLD, 1e-6)) * 100), 1)


VERDICT_LABELS = {
    "match": ("Certain signature match", "success"), # Tuple = User label and Streamlit message style
    "uncertain": ("Uncertain — try another sample", "warning"),
    "no_match": ("Certain signature mismatch", "error")
}
