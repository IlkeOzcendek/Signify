"""Evaluate technical routing between general v1 and UTSig head on held-out people."""

import json
import math
from pathlib import Path

import cv2
import pandas as pd
import torch

from scripts.evaluate_utsig_domain_protocol import protocol_rows, registered_verdict, verdict
from scripts.train_transfer_head import PairHead
from scripts.train_utsig_domain_head import records
from src.preprocessing.signature_dataset import split_person_ids


V1_PREDICTIONS = Path("outputs/realistic_evaluation_20260712_005559/predictions.csv")
FEATURES = Path("outputs/transfer_head_20260711_235238/features.pt")
HEAD = Path("outputs/utsig_domain_head_20260712_031710/best_head.pth")


def high_resolution_mono(path):
    image = cv2.imread(str(path))
    if image is None:
        return False
    height, width = image.shape[:2]
    saturation = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)[:, :, 1].mean()
    return height >= 450 and width >= 650 and saturation <= 1.5


def route(paths):
    flags = [high_resolution_mono(path) for path in paths]
    return "utsig" if sum(flags) >= math.ceil(len(flags) * 0.60) else "general"


def summarize(rows):
    decided = [row for row in rows if row["verdict"] != "uncertain"]
    genuine = [row for row in rows if row["expected"] == "match"]
    forgery = [row for row in rows if row["expected"] == "no_match"]
    return {
        "total": len(rows),
        "coverage": len(decided) / len(rows),
        "uncertain_rate": 1 - len(decided) / len(rows),
        "decided_accuracy": sum(r["verdict"] == r["expected"] for r in decided) / max(len(decided), 1),
        "far": sum(r["verdict"] == "match" for r in forgery) / max(len(forgery), 1),
        "frr": sum(r["verdict"] == "no_match" for r in genuine) / max(len(genuine), 1),
        "routed_utsig": sum(r["engine"] == "utsig" for r in rows) / len(rows),
    }


def main():
    predictions = pd.read_csv(V1_PREDICTIONS)
    lookup = {
        (row["mode"], row["query"]): row["verdict"]
        for row in predictions.to_dict("records")
    }
    features = torch.load(FEATURES, map_location="cpu")
    head = PairHead().cpu().eval()
    head.load_state_dict(torch.load(HEAD, map_location="cpu"))
    _, _, test_ids = split_person_ids(seed=42)
    rows = []
    for source, ids in (
        ("CEDAR", [value for value in test_ids if value < 1000]),
        ("UTSig", [value for value in test_ids if value >= 2000]),
    ):
        genuine, forgery = records(ids)
        manual_values, registered_values = protocol_rows(ids, head, features)
        manual_index = registered_index = 0
        for person_id in sorted(genuine):
            references = genuine[person_id][:10]
            queries = [(path, "match") for path in genuine[person_id][10:]]
            queries += [(path, "no_match") for path in forgery.get(person_id, [])]
            for query, expected in queries:
                for mode in ("manual", "registered"):
                    paths = [references[0], query] if mode == "manual" else [*references, query]
                    engine = route(paths)
                    if mode == "manual":
                        values, _ = manual_values[manual_index]
                        prediction = verdict(values, 0.37, 0.76)
                    else:
                        values, _ = registered_values[registered_index]
                        prediction = registered_verdict(values, 0.45, 0.61)
                    if engine == "utsig":
                        decision = {1: "match", 0: "no_match", None: "uncertain"}[prediction]
                    else:
                        decision = lookup[(mode, str(query))]
                    rows.append({
                        "source": source, "mode": mode, "query": str(query),
                        "expected": expected, "verdict": decision, "engine": engine,
                    })
                manual_index += 1
                registered_index += 1
    report = {}
    for mode in ("manual", "registered"):
        report[mode] = {
            source: summarize([row for row in rows if row["mode"] == mode and row["source"] == source])
            for source in ("CEDAR", "UTSig")
        }
        report[mode]["mixed"] = summarize([row for row in rows if row["mode"] == mode])
    Path("outputs/mixed_router_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
