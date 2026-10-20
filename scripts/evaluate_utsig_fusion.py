"""Select binary/gray UTSig fusion and bands on validation people only."""

import json
from pathlib import Path

import torch

from scripts.evaluate_utsig_domain_protocol import calibrate, summarize
from scripts.train_transfer_head import PairHead
from scripts.train_utsig_domain_head import records, utsig_ids
from src.preprocessing.signature_dataset import split_person_ids


BINARY_RUN = Path("outputs/utsig_domain_head_20260712_031710")
GRAY_RUN = Path("outputs/utsig_domain_head_gray_20260712_032559")
BINARY_FEATURES = Path("outputs/transfer_head_20260711_235238/features.pt")
GRAY_FEATURES = Path("outputs/utsig_gray_features.pt")
OUTPUT = Path("outputs/utsig_fusion_report.json")


def load(run, features_path):
    head = PairHead().cpu().eval()
    head.load_state_dict(torch.load(run / "best_head.pth", map_location="cpu"))
    return head, torch.load(features_path, map_location="cpu")


def probability(model, first, second):
    head, features = model
    with torch.no_grad():
        return torch.sigmoid(head(
            features[str(first)].unsqueeze(0),
            features[str(second)].unsqueeze(0),
        )).item()


def raw_rows(ids, binary, gray):
    genuine, forgery = records(ids)
    manual, registered = [], []
    for person_id in sorted(genuine):
        references = genuine[person_id][:10]
        queries = [(path, 1) for path in genuine[person_id][10:]]
        queries += [(path, 0) for path in forgery.get(person_id, [])]
        for query, label in queries:
            pair_values = [
                (
                    probability(binary, reference, query),
                    probability(gray, reference, query),
                )
                for reference in references
            ]
            manual.append((pair_values[0], label))
            registered.append((pair_values, label))
    return manual, registered


def fuse(rows, gray_weight, registered=False):
    if registered:
        return [
            ([gray_weight * gray + (1 - gray_weight) * binary
              for binary, gray in values], label)
            for values, label in rows
        ]
    return [
        (gray_weight * values[1] + (1 - gray_weight) * values[0], label)
        for values, label in rows
    ]


def choose(validation_rows, registered=False):
    best = None
    for step in range(11):
        weight = step / 10
        fused = fuse(validation_rows, weight, registered=registered)
        bands = calibrate(fused, registered=registered)
        if bands is None:
            continue
        coverage, neg_frr, low, high, result = bands
        candidate = (coverage, neg_frr, weight, low, high, result)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best


def main():
    binary = load(BINARY_RUN, BINARY_FEATURES)
    gray = load(GRAY_RUN, GRAY_FEATURES)
    _, validation_ids, test_ids = split_person_ids(seed=42)
    validation = raw_rows(utsig_ids(validation_ids), binary, gray)
    test = raw_rows(utsig_ids(test_ids), binary, gray)
    report = {}
    for index, (mode, registered) in enumerate((("manual", False), ("registered", True))):
        best = choose(validation[index], registered=registered)
        _, _, weight, low, high, validation_result = best
        report[mode] = {
            "gray_weight": weight,
            "bands": {"different_max": low, "same_min": high},
            "validation": validation_result,
            "test": summarize(
                fuse(test[index], weight, registered=registered),
                low, high, registered=registered,
            ),
        }
    OUTPUT.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
