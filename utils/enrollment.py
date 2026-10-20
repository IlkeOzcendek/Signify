"""Make robust multi-reference decisions for registered-person verification.""" # Docstring = Official description stored for this module

import math
import statistics


def decide_registered_person(
    distances, threshold, confident_diff, confident_same=None
):
    """Return a verdict using reference consensus instead of one lucky match.""" # Consensus = Several references must support the same decision

    if not distances:
        raise ValueError("At least one reference distance is required.")

    ordered = sorted(float(value) for value in distances) # ordered = Reference distances converted to numbers and sorted from low to high
    count = len(ordered) # Number of registered references used in the decision

    if confident_same is None:
        confident_same = threshold # Uses the binary threshold when no separate confident-match boundary exists

    if count < 5:
        decision_distance = statistics.median(ordered) # Median = Middle distance ------> less affected by one unusual reference

        if decision_distance < confident_same:
            verdict = "match"
        elif decision_distance > confident_diff:
            verdict = "no_match"
        else:
            verdict = "uncertain"

        return {
            "verdict": verdict, # Final text decision
            "decision_distance": decision_distance, # Median distance representing all available references
            "support_count": sum(value < threshold for value in ordered), # References below the match threshold
            "required_support": math.ceil(count / 2), # Half of references rounded upward
            "method": "median_fallback", # Small groups rely mainly on their median
        }

    required_support = max(3, math.ceil(count * 0.60)) # Needs at least three votes and 60% of all references
    support_count = sum(value < threshold for value in ordered) # References whose distance votes for a match
    decision_distance = statistics.median(ordered)

    if support_count >= required_support and decision_distance < confident_same:
        verdict = "match"
    elif decision_distance > confident_diff:
        verdict = "no_match"
    else:
        verdict = "uncertain"

    return {
        "verdict": verdict,
        "decision_distance": decision_distance,
        "support_count": support_count,
        "required_support": required_support,
        "method": "majority_median", # Final match needs both majority support and a confidently low median
    }
