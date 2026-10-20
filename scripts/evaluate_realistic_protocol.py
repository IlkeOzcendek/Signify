"""Evaluate the exact application pipeline with a realistic enrollment protocol."""

import csv
import json
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids
from utils.enrollment import decide_registered_person
from utils.helpers import THRESHOLD, T_DIFF, T_SAME, classify_verdict
from verify import get_model, device


BATCH_SIZE = 32


def source_name(person_id):
    if person_id < 1000:
        return "CEDAR"
    if person_id < 2000:
        return "custom_selected"
    return "UTSig"


def encode_paths(dataset, paths):
    model = get_model()
    embeddings = {}
    ink_ratios = {}
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = torch.stack([dataset.load_image(path) for path in batch_paths])
            ink = (images < 0.5).float().mean(dim=(1, 2, 3))
            outputs = model.forward_once(images.to(device)).cpu()
            for path, output, ratio in zip(batch_paths, outputs, ink):
                embeddings[str(path)] = output
                ink_ratios[str(path)] = float(ratio)
            if start % (BATCH_SIZE * 10) == 0:
                print(f"encoded={min(start + BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)
    return embeddings, ink_ratios


def distance(embeddings, path_1, path_2):
    return float(F.pairwise_distance(
        embeddings[str(path_1)].unsqueeze(0),
        embeddings[str(path_2)].unsqueeze(0),
    ).item())


def summarize(rows, mode=None, source=None, thickness=None):
    selected = [row for row in rows if (mode is None or row["mode"] == mode)]
    selected = [row for row in selected if (source is None or row["source"] == source)]
    selected = [row for row in selected if (thickness is None or row["thickness"] == thickness)]
    if not selected:
        return None
    decided = [row for row in selected if row["verdict"] != "uncertain"]
    correct = sum(row["verdict"] == row["expected"] for row in decided)
    genuine = [row for row in selected if row["expected"] == "match"]
    forgery = [row for row in selected if row["expected"] == "no_match"]
    false_reject = sum(row["verdict"] == "no_match" for row in genuine)
    false_accept = sum(row["verdict"] == "match" for row in forgery)
    return {
        "total": len(selected),
        "coverage": len(decided) / len(selected),
        "decided_accuracy": correct / max(len(decided), 1),
        "overall_correct": sum(row["verdict"] == row["expected"] for row in selected) / len(selected),
        "uncertain_rate": sum(row["verdict"] == "uncertain" for row in selected) / len(selected),
        "far": false_accept / max(len(forgery), 1),
        "frr": false_reject / max(len(genuine), 1),
        "genuine_count": len(genuine),
        "forgery_count": len(forgery),
    }


def main():
    _, _, test_ids = split_person_ids()
    dataset = SignatureDataset(person_ids=test_ids, augment=False, seed=42)
    originals = defaultdict(list)
    forgeries = defaultdict(list)
    for path, person_id in dataset.original_records:
        originals[person_id].append(path)
    for path, person_id in dataset.forged_records:
        forgeries[person_id].append(path)
    for values in originals.values():
        values.sort(key=str)
    for values in forgeries.values():
        values.sort(key=str)

    all_paths = sorted(
        {path for values in originals.values() for path in values}
        | {path for values in forgeries.values() for path in values},
        key=str,
    )
    embeddings, ink_ratios = encode_paths(dataset, all_paths)

    genuine_ink = sorted(ink_ratios[str(path)] for values in originals.values() for path in values)
    low_cut = genuine_ink[len(genuine_ink) // 3]
    high_cut = genuine_ink[(2 * len(genuine_ink)) // 3]

    def thickness(path):
        ratio = ink_ratios[str(path)]
        return "thin" if ratio <= low_cut else "thick" if ratio >= high_cut else "normal"

    rows = []
    for person_id in sorted(originals):
        genuine = originals[person_id]
        forged = forgeries.get(person_id, [])
        if len(genuine) < 2:
            continue
        enrollment_count = min(10, max(1, len(genuine) // 2))
        enrollment = genuine[:enrollment_count]
        genuine_queries = genuine[enrollment_count:]

        for expected, queries in (("match", genuine_queries), ("no_match", forged)):
            for query in queries:
                distances = [distance(embeddings, reference, query) for reference in enrollment]
                registered = decide_registered_person(
                    distances, THRESHOLD, T_DIFF, T_SAME
                )
                rows.append({
                    "mode": "registered",
                    "source": source_name(person_id),
                    "person_id": person_id,
                    "query": str(query),
                    "expected": expected,
                    "verdict": registered["verdict"],
                    "distance": registered["decision_distance"],
                    "support": registered["support_count"],
                    "required_support": registered["required_support"],
                    "references": len(enrollment),
                    "ink_ratio": ink_ratios[str(query)],
                    "thickness": thickness(query),
                })

                # Manual mode uses a fixed first genuine reference.
                manual_distance = distance(embeddings, enrollment[0], query)
                rows.append({
                    "mode": "manual",
                    "source": source_name(person_id),
                    "person_id": person_id,
                    "query": str(query),
                    "expected": expected,
                    "verdict": classify_verdict(manual_distance),
                    "distance": manual_distance,
                    "support": "",
                    "required_support": "",
                    "references": 1,
                    "ink_ratio": ink_ratios[str(query)],
                    "thickness": thickness(query),
                })

    output_dir = Path("outputs") / f"realistic_evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True)
    with open(output_dir / "predictions.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "configuration": {
            "threshold": THRESHOLD,
            "confident_same": T_SAME,
            "confident_diff": T_DIFF,
            "test_people": test_ids,
            "thin_ink_cut": low_cut,
            "thick_ink_cut": high_cut,
        },
        "overall": {},
        "by_source": {},
        "by_thickness": {},
    }
    for mode in ("manual", "registered"):
        report["overall"][mode] = summarize(rows, mode=mode)
        report["by_source"][mode] = {
            source: summarize(rows, mode=mode, source=source)
            for source in ("CEDAR", "custom_selected", "UTSig")
        }
        report["by_thickness"][mode] = {
            group: summarize(rows, mode=mode, thickness=group)
            for group in ("thin", "normal", "thick")
        }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
