"""Evaluate an archive using its <person><T|F><sample> naming scheme."""

import csv
import json
import os
import re
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import torch
import torch.nn.functional as F

from utils.enrollment import decide_registered_person
from utils.helpers import THRESHOLD, T_DIFF, T_SAME, classify_verdict
from verify import device, get_model, load_image


ARCHIVE = Path(os.environ.get("SIGNIFY_ARCHIVE_DATASET", "dataset/archive/Signature Images"))
BATCH_SIZE = 32


def parse_name(path):
    match = re.match(r"^(\d{1,2})([TF])(\d+)\.(?:jpe?g|png)$", path.name, re.I)
    if not match:
        return None
    person, kind, sample = match.groups()
    return person.zfill(2), kind.upper(), int(sample)


def encode(paths):
    model = get_model()
    result = {}
    ink_ratios = {}
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = torch.cat([load_image(str(path)) for path in batch_paths], dim=0)
            outputs = model.forward_once(images).cpu()
            inks = (images.cpu() < 0.5).float().mean(dim=(1, 2, 3))
            for path, output, ink in zip(batch_paths, outputs, inks):
                result[str(path)] = output
                ink_ratios[str(path)] = float(ink)
            if start % (BATCH_SIZE * 10) == 0:
                print(f"encoded={min(start + BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)
    return result, ink_ratios


def distance(embeddings, path_1, path_2):
    return float(F.pairwise_distance(
        embeddings[str(path_1)].unsqueeze(0),
        embeddings[str(path_2)].unsqueeze(0),
    ).item())


def metrics(rows):
    decided = [row for row in rows if row["verdict"] != "uncertain"]
    genuine = [row for row in rows if row["expected"] == "match"]
    forgery = [row for row in rows if row["expected"] == "no_match"]
    return {
        "total": len(rows),
        "genuine": len(genuine),
        "forgery": len(forgery),
        "coverage": len(decided) / max(len(rows), 1),
        "uncertain_rate": sum(row["verdict"] == "uncertain" for row in rows) / max(len(rows), 1),
        "decided_accuracy": sum(row["verdict"] == row["expected"] for row in decided) / max(len(decided), 1),
        "far": sum(row["verdict"] == "match" for row in forgery) / max(len(forgery), 1),
        "frr": sum(row["verdict"] == "no_match" for row in genuine) / max(len(genuine), 1),
        "genuine_match": sum(row["verdict"] == "match" for row in genuine) / max(len(genuine), 1),
        "forgery_rejected": sum(row["verdict"] == "no_match" for row in forgery) / max(len(forgery), 1),
    }


def main():
    if not ARCHIVE.is_dir():
        raise SystemExit("Set SIGNIFY_ARCHIVE_DATASET to the signature archive directory.")
    people = defaultdict(lambda: {"T": [], "F": []})
    ignored = []
    for path in sorted(ARCHIVE.iterdir(), key=lambda value: value.name):
        parsed = parse_name(path)
        if parsed is None:
            if path.is_file() and path.name != ".DS_Store":
                ignored.append(path.name)
            continue
        person, kind, sample = parsed
        people[person][kind].append((sample, path))
    for kinds in people.values():
        for kind in kinds:
            kinds[kind] = [path for _, path in sorted(kinds[kind])]

    usable_people = {
        person: kinds for person, kinds in people.items()
        if len(kinds["T"]) >= 11 and kinds["F"]
    }
    paths = sorted(
        {path for kinds in usable_people.values() for kind in ("T", "F") for path in kinds[kind]},
        key=str,
    )
    embeddings, ink_ratios = encode(paths)

    rows = []
    for person, kinds in sorted(usable_people.items()):
        enrollment = kinds["T"][:10]
        queries = [("match", path) for path in kinds["T"][10:]]
        queries += [("no_match", path) for path in kinds["F"]]
        for expected, query in queries:
            distances = [distance(embeddings, reference, query) for reference in enrollment]
            decision = decide_registered_person(distances, THRESHOLD, T_DIFF, T_SAME)
            rows.append({
                "mode": "registered",
                "person": person,
                "query": query.name,
                "expected": expected,
                "verdict": decision["verdict"],
                "distance": decision["decision_distance"],
                "support": decision["support_count"],
                "required_support": decision["required_support"],
                "ink_ratio": ink_ratios[str(query)],
            })
            manual_distance = distance(embeddings, enrollment[0], query)
            rows.append({
                "mode": "manual",
                "person": person,
                "query": query.name,
                "expected": expected,
                "verdict": classify_verdict(manual_distance),
                "distance": manual_distance,
                "support": "",
                "required_support": "",
                "ink_ratio": ink_ratios[str(query)],
            })

    output_dir = Path("outputs") / f"archive_evaluation_{datetime.now():%Y%m%d_%H%M%S}"
    output_dir.mkdir(parents=True)
    with open(output_dir / "predictions.csv", "w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    report = {
        "configuration": {
            "archive": str(ARCHIVE),
            "people_found": len(people),
            "people_tested": len(usable_people),
            "excluded_people": sorted(set(people) - set(usable_people)),
            "ignored_files": ignored,
            "enrollment_per_person": 10,
            "threshold": THRESHOLD,
            "confident_same": T_SAME,
            "confident_diff": T_DIFF,
        },
        "manual": metrics([row for row in rows if row["mode"] == "manual"]),
        "registered": metrics([row for row in rows if row["mode"] == "registered"]),
        "registered_by_person": {
            person: metrics([
                row for row in rows
                if row["mode"] == "registered" and row["person"] == person
            ])
            for person in sorted(usable_people)
        },
    }
    (output_dir / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps(report, indent=2), flush=True)
    print(f"output={output_dir}", flush=True)


if __name__ == "__main__":
    main()
