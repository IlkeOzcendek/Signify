"""Audit/evaluate the copied signature dataset without trusting sample duplicates."""

import json
import math
import os
import re
from collections import defaultdict
from pathlib import Path

import cv2
import torch
import torch.nn.functional as F

from scripts.train_transfer_head import PairHead
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.archive_grayscale_v2 import load_image as load_gray
from utils.enrollment import decide_registered_person
from utils.helpers import THRESHOLD, T_DIFF, T_SAME, classify_verdict
from verify import get_model, load_image
from verify_archive_v2 import (
    archive_verdict, decide_archive_registered,
    get_archive_models, GRAY_WEIGHT,
)
from verify_utsig_v3 import classify as utsig_verdict, decide_registered as decide_utsig_registered


ROOT = Path(os.environ.get("SIGNIFY_SKILLED_DATASET", "dataset/skilled_forgery"))
BATCH = 32


def parse():
    people = defaultdict(lambda: {"real": [], "forge": []})
    ignored = []
    for path in ROOT.rglob("*"):
        if not path.is_file() or path.suffix.lower() != ".png":
            continue
        match = re.match(r"^(\d{3})(\d{2})(\d{3})$", path.stem)
        if not match:
            ignored.append(str(path))
            continue
        writer, sample, target = match.groups()
        label = "forge" if path.parent.name == "forge" else "real"
        # dataset1..4 are overlapping alternative partitions, not one additive
        # population. Keep their identities separate during the audit.
        dataset_name = path.parents[1].name
        person = f"{dataset_name}:{target if label == 'forge' else writer}"
        people[person][label].append(path)
    for kinds in people.values():
        for values in kinds.values():
            values.sort(key=str)
    return dict(people), ignored


def batch_features(paths):
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    legacy = get_model().to(device).eval()
    mobile = MobileNetSignatureEncoder(projection=False, freeze_backbone=True).to(device).eval()
    legacy_values, binary_values, gray_values = {}, {}, {}
    with torch.no_grad():
        for start in range(0, len(paths), BATCH):
            batch_paths = paths[start:start + BATCH]
            binary = torch.cat([load_image(path) for path in batch_paths]).to(device)
            gray = torch.stack([load_gray(path) for path in batch_paths]).to(device)
            lv = legacy.forward_once(binary).cpu()
            bv = mobile.forward_once(binary).cpu()
            gv = mobile.forward_once(gray).cpu()
            for path, a, b, g in zip(batch_paths, lv, bv, gv):
                legacy_values[str(path)] = a
                binary_values[str(path)] = b
                gray_values[str(path)] = g
            print(f"features={min(start+BATCH,len(paths))}/{len(paths)}", flush=True)
    return legacy_values, binary_values, gray_values


def head_probability(head, features, first, second):
    with torch.no_grad():
        return torch.sigmoid(head(
            features[str(first)].unsqueeze(0), features[str(second)].unsqueeze(0)
        )).item()


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
    }


def main():
    if not ROOT.is_dir():
        raise SystemExit("Set SIGNIFY_SKILLED_DATASET to the skilled-forgery dataset directory.")
    people, ignored = parse()
    usable = {person: kinds for person, kinds in people.items() if len(kinds["real"]) == 5 and len(kinds["forge"]) == 5}
    paths = sorted({p for kinds in usable.values() for values in kinds.values() for p in values}, key=str)
    legacy, binary, gray = batch_features(paths)
    _, archive_binary_head, archive_gray_head = get_archive_models()
    utsig_head = PairHead().cpu().eval()
    utsig_head.load_state_dict(torch.load("models/utsig_v3_binary_head.pth", map_location="cpu"))
    rows = []
    for person, kinds in sorted(usable.items()):
        references = kinds["real"][:3]
        queries = [(path, "match") for path in kinds["real"][3:]]
        queries += [(path, "no_match") for path in kinds["forge"]]
        for query, expected in queries:
            v1_distances, v2_probs, v3_probs = [], [], []
            for reference in references:
                v1_distances.append(float(F.pairwise_distance(
                    legacy[str(reference)].unsqueeze(0), legacy[str(query)].unsqueeze(0)
                )))
                binary_prob = head_probability(archive_binary_head, binary, reference, query)
                gray_prob = head_probability(archive_gray_head, gray, reference, query)
                v2_probs.append(GRAY_WEIGHT * gray_prob + (1 - GRAY_WEIGHT) * binary_prob)
                v3_probs.append(head_probability(utsig_head, binary, reference, query))
            decisions = {
                ("v1", "manual"): classify_verdict(v1_distances[0]),
                ("v1", "registered"): decide_registered_person(v1_distances, THRESHOLD, T_DIFF, T_SAME)["verdict"],
                ("v2", "manual"): archive_verdict(v2_probs[0]),
                ("v2", "registered"): decide_archive_registered(v2_probs)["verdict"],
                ("v3", "manual"): utsig_verdict(v3_probs[0]),
                ("v3", "registered"): decide_utsig_registered(v3_probs)["verdict"],
            }
            for (engine, mode), decision in decisions.items():
                rows.append({"person": person, "query": query.name, "expected": expected, "engine": engine, "mode": mode, "verdict": decision})
    report = {
        "people_found": len(people), "people_usable": len(usable),
        "images": len(paths), "ignored": ignored,
        "metrics": {
            mode: {engine: summarize([r for r in rows if r["mode"] == mode and r["engine"] == engine]) for engine in ("v1", "v2", "v3")}
            for mode in ("manual", "registered")
        },
        "by_dataset": {
            dataset_name: {
                mode: {
                    engine: summarize([
                        r for r in rows
                        if r["mode"] == mode and r["engine"] == engine
                        and r["person"].startswith(dataset_name + ":")
                    ])
                    for engine in ("v1", "v2", "v3")
                }
                for mode in ("manual", "registered")
            }
            for dataset_name in ("dataset1", "dataset2", "dataset3", "dataset4")
        },
    }
    output = Path("outputs/signature_copy_dataset_report.json")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__":
    main()
