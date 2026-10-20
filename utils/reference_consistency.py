import statistics

from utils.helpers import THRESHOLD, T_DIFF
from verify import verify_signature


def analyze_reference_consistency(reference_paths):
    results = [] # Quality summary produced for every registered reference

    for reference in reference_paths:
        other_distances = [] # Distances from the current reference to every other reference

        for other in reference_paths:
            if other == reference:
                continue # A reference must not be compared with itself

            _, distance = verify_signature(
                str(reference), str(other), threshold = THRESHOLD
            )

            other_distances.append(distance)

        median_distance = (
            statistics.median(other_distances) if other_distances else 0.0
        ) # Median = Middle distance ------> less affected by one unusual comparison

        matching_count = sum(value < THRESHOLD for value in other_distances) # Other references close enough to support a match
        minimum_matches = max(1, len(other_distances) // 2) # Requires at least half of the available comparisons and never less than one

        is_outlier = (
            median_distance > T_DIFF or matching_count < minimum_matches
        ) # Outlier = Reference far from the group or supported by too few other references

        results.append({
            "reference": reference.name, # File name of the reference being analyzed
            "median_distance": median_distance, # Stable distance representing all comparisons
            "matching_count": matching_count, # Number of references voting for a match
            "compared_count": len(other_distances), # Total number of comparisons made
            "is_outlier": is_outlier # True means this reference may not be consistent with the registered set
        })

    return results
