"""Check whether abstention can rescue the isolated skilled-forgery head."""

import json
from pathlib import Path

import torch

from scripts.train_skilled_forgery_head import (
    load_people, make_pairs, metrics, probabilities, split, tensors,
)
from scripts.train_transfer_head import PairHead


RUN = Path("outputs/skilled_head_20260712_050415")


def abstention(probs, labels, low, high):
    predictions = torch.full_like(labels, -1)
    predictions[probs <= low] = 0
    predictions[probs >= high] = 1
    decided = predictions >= 0
    positive, negative = labels == 1, labels == 0
    return {
        "coverage": float(decided.float().mean()),
        "uncertain_rate": float((~decided).float().mean()),
        "decided_accuracy": float((predictions[decided] == labels[decided]).float().mean()) if decided.any() else 0,
        "far": float(((predictions == 1) & negative).sum() / max(int(negative.sum()), 1)),
        "frr": float(((predictions == 0) & positive).sum() / max(int(positive.sum()), 1)),
    }


def choose(probs, labels):
    best = None
    for low_i in range(5, 96):
        for high_i in range(low_i + 1, 96):
            result = abstention(probs, labels, low_i / 100, high_i / 100)
            if result["far"] > 0.10 or result["frr"] > 0.20:
                continue
            if result["decided_accuracy"] < 0.80:
                continue
            candidate = (result["coverage"], low_i / 100, high_i / 100, result)
            if best is None or candidate[0] > best[0]:
                best = candidate
    return best


def main():
    people = load_people()
    _, validation_ids, test_ids = split(people)
    validation_rows = make_pairs(people, validation_ids, cap=False)
    test_rows = make_pairs(people, test_ids, cap=False)
    report = {}
    for name in ("binary", "gray"):
        features = torch.load(RUN / f"{name}_features.pt", map_location="cpu")
        head = PairHead().cpu().eval()
        head.load_state_dict(torch.load(RUN / f"{name}_head.pth", map_location="cpu"))
        validation_probs, validation_labels = probabilities(head, tensors(validation_rows, features))
        test_probs, test_labels = probabilities(head, tensors(test_rows, features))
        best = choose(validation_probs, validation_labels)
        if best is None:
            report[name] = {"error": "No bands satisfy all safety constraints"}
        else:
            _, low, high, validation_result = best
            report[name] = {
                "bands": {"different_max": low, "same_min": high},
                "validation": validation_result,
                "test": abstention(test_probs, test_labels, low, high),
            }
    (RUN / "calibration.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
