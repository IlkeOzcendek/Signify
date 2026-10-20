"""Calibrate UTSig head abstention bands on validation people, test once on held-out people."""

import argparse
import json
import math
import statistics
from pathlib import Path

import torch

from scripts.train_transfer_head import PairHead
from scripts.train_utsig_domain_head import FEATURES, records, utsig_ids
from src.preprocessing.signature_dataset import split_person_ids


RUN = Path("outputs/utsig_domain_head_20260712_031710")
TARGET_DECIDED_ACCURACY = 0.80


def probability(head, features, first, second):
    with torch.no_grad():
        return torch.sigmoid(head(
            features[str(first)].unsqueeze(0),
            features[str(second)].unsqueeze(0),
        )).item()


def protocol_rows(ids, head, features):
    genuine, forgery = records(ids)
    manual, registered = [], []
    for person_id in sorted(genuine):
        enrollment = genuine[person_id][:10]
        queries = [(path, 1) for path in genuine[person_id][10:]]
        queries += [(path, 0) for path in forgery.get(person_id, [])]
        for query, label in queries:
            probabilities = [
                probability(head, features, reference, query)
                for reference in enrollment
            ]
            manual.append((probabilities[0], label))
            registered.append((probabilities, label))
    return manual, registered


def verdict(value, low, high):
    if value >= high:
        return 1
    if value <= low:
        return 0
    return None


def registered_verdict(values, low, high):
    median = statistics.median(values)
    required = max(1, math.ceil(len(values) * 0.60))
    support = sum(value >= high for value in values)
    if support >= required and median >= high:
        return 1
    if median <= low:
        return 0
    return None


def summarize(rows, low, high, registered=False):
    decisions = []
    for values, label in rows:
        prediction = (
            registered_verdict(values, low, high)
            if registered else verdict(values, low, high)
        )
        decisions.append((prediction, label))
    decided = [(prediction, label) for prediction, label in decisions if prediction is not None]
    genuine = [(prediction, label) for prediction, label in decisions if label == 1]
    forgery = [(prediction, label) for prediction, label in decisions if label == 0]
    return {
        "total": len(decisions),
        "coverage": len(decided) / len(decisions),
        "uncertain_rate": 1 - len(decided) / len(decisions),
        "decided_accuracy": sum(p == y for p, y in decided) / max(len(decided), 1),
        "far": sum(p == 1 for p, _ in forgery) / max(len(forgery), 1),
        "frr": sum(p == 0 for p, _ in genuine) / max(len(genuine), 1),
    }


def calibrate(rows, registered=False):
    candidates = [index / 100 for index in range(5, 96)]
    best = None
    for low in candidates:
        for high in candidates:
            if low >= high:
                continue
            result = summarize(rows, low, high, registered=registered)
            if result["decided_accuracy"] < TARGET_DECIDED_ACCURACY:
                continue
            # Protect against accepting forgeries; then maximize useful coverage.
            if result["far"] > 0.10:
                continue
            if result["frr"] > 0.20:
                continue
            candidate = (result["coverage"], -result["frr"], low, high, result)
            if best is None or candidate[:2] > best[:2]:
                best = candidate
    return best


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", type=Path, default=RUN)
    parser.add_argument("--features", type=Path, default=FEATURES)
    args = parser.parse_args()
    features = torch.load(args.features, map_location="cpu")
    head = PairHead().cpu().eval()
    head.load_state_dict(torch.load(args.run / "best_head.pth", map_location="cpu"))
    _, validation_ids, test_ids = split_person_ids(seed=42)
    validation_ids, test_ids = utsig_ids(validation_ids), utsig_ids(test_ids)
    validation_manual, validation_registered = protocol_rows(validation_ids, head, features)
    test_manual, test_registered = protocol_rows(test_ids, head, features)
    report = {}
    for mode, validation, test, is_registered in (
        ("manual", validation_manual, test_manual, False),
        ("registered", validation_registered, test_registered, True),
    ):
        best = calibrate(validation, registered=is_registered)
        if best is None:
            report[mode] = {"error": "no bands satisfy constraints"}
            continue
        _, _, low, high, validation_result = best
        report[mode] = {
            "bands": {"different_max": low, "same_min": high},
            "validation": validation_result,
            "test": summarize(test, low, high, registered=is_registered),
        }
    (args.run / "protocol_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
