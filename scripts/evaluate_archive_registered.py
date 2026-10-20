"""Evaluate archive v2 using the same multi-reference protocol as the app."""

from collections import Counter

import torch

from scripts.finetune_archive_short import load_people, split_people
from scripts.train_archive_transfer_v2 import PairHead
from verify_archive_v2 import archive_verdict, decide_archive_registered


BINARY_RUN = "outputs/archive_transfer_v2_20260712_022322"
GRAY_RUN = "outputs/archive_transfer_v2_20260712_023115"


def load_run(path):
    features = torch.load(f"{path}/features.pt", map_location="cpu")
    head = PairHead().cpu().eval()
    head.load_state_dict(torch.load(f"{path}/best_head.pth", map_location="cpu"))
    return features, head


def pair_probability(reference, query, binary, gray):
    binary_features, binary_head = binary
    gray_features, gray_head = gray
    with torch.no_grad():
        bp = torch.sigmoid(binary_head(
            binary_features[str(reference)].unsqueeze(0),
            binary_features[str(query)].unsqueeze(0),
        )).item()
        gp = torch.sigmoid(gray_head(
            gray_features[str(reference)].unsqueeze(0),
            gray_features[str(query)].unsqueeze(0),
        )).item()
    return 0.75 * gp + 0.25 * bp


def evaluate(people, person_ids, binary, gray):
    counts = Counter()
    rows = []
    for person in person_ids:
        references = people[person]["T"][:10]
        queries = [(path, "match") for path in people[person]["T"][10:]]
        queries += [(path, "no_match") for path in people[person]["F"]]
        for query, expected in queries:
            probabilities = [
                pair_probability(reference, query, binary, gray)
                for reference in references
            ]
            decision = decide_archive_registered(probabilities)
            verdict = decision["verdict"]
            counts[f"expected_{expected}"] += 1
            counts[f"verdict_{verdict}"] += 1
            if verdict == "uncertain":
                counts["uncertain"] += 1
            elif verdict == expected:
                counts["correct_decided"] += 1
            else:
                counts["wrong_decided"] += 1
            rows.append((person, query.name, expected, verdict, decision["probability"]))
    total = len(rows)
    decided = counts["correct_decided"] + counts["wrong_decided"]
    return {
        "people": len(person_ids),
        "queries": total,
        "coverage": decided / total,
        "uncertain_rate": counts["uncertain"] / total,
        "decided_accuracy": counts["correct_decided"] / max(decided, 1),
        "counts": dict(counts),
    }


def evaluate_manual(people, person_ids, binary, gray):
    """Mirror the app: one uploaded reference versus one questioned image."""
    counts = Counter()
    per_class = Counter()
    for person in person_ids:
        reference = people[person]["T"][0]
        queries = [(path, "match") for path in people[person]["T"][10:]]
        queries += [(path, "no_match") for path in people[person]["F"]]
        for query, expected in queries:
            verdict = archive_verdict(
                pair_probability(reference, query, binary, gray)
            )
            counts[f"expected_{expected}"] += 1
            counts[f"verdict_{verdict}"] += 1
            per_class[f"{expected}_{verdict}"] += 1
            if verdict == "uncertain":
                counts["uncertain"] += 1
            elif verdict == expected:
                counts["correct_decided"] += 1
            else:
                counts["wrong_decided"] += 1
    total = counts["expected_match"] + counts["expected_no_match"]
    decided = counts["correct_decided"] + counts["wrong_decided"]
    return {
        "people": len(person_ids),
        "queries": total,
        "coverage": decided / total,
        "uncertain_rate": counts["uncertain"] / total,
        "decided_accuracy": counts["correct_decided"] / max(decided, 1),
        "genuine_match_rate": per_class["match_match"] / counts["expected_match"],
        "genuine_uncertain_rate": per_class["match_uncertain"] / counts["expected_match"],
        "genuine_false_reject_rate": per_class["match_no_match"] / counts["expected_match"],
        "forgery_reject_rate": per_class["no_match_no_match"] / counts["expected_no_match"],
        "forgery_uncertain_rate": per_class["no_match_uncertain"] / counts["expected_no_match"],
        "false_accept_rate": per_class["no_match_match"] / counts["expected_no_match"],
        "counts": dict(counts),
    }


def main():
    people = load_people()
    binary = load_run(BINARY_RUN)
    gray = load_run(GRAY_RUN)
    train_ids, validation_ids, test_ids = split_people(people)
    for name, ids in (
        ("validation", validation_ids),
        ("test", test_ids),
        ("all", sorted(people)),
    ):
        print(name, "registered", evaluate(people, ids, binary, gray))
        print(name, "manual", evaluate_manual(people, ids, binary, gray))


if __name__ == "__main__":
    main()
