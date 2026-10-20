"""Compare fixed-query performance for 1/2/3/5/10 enrollment references."""

import json
from collections import defaultdict
from pathlib import Path

import torch
import torch.nn.functional as F

from scripts.evaluate_archive_registered import load_run, pair_probability as archive_probability
from scripts.finetune_archive_short import load_people as load_archive_people, split_people
from scripts.train_transfer_head import PairHead
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids
from utils.enrollment import decide_registered_person
from utils.helpers import THRESHOLD, T_DIFF, T_SAME
from verify import get_model, load_image
from verify_archive_v2 import decide_archive_registered
from verify_utsig_v3 import decide_registered as decide_utsig_registered


COUNTS = (1, 2, 3, 5, 10)


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
        "genuine_match": sum(r["verdict"] == "match" for r in genuine) / max(len(genuine), 1),
        "forgery_reject": sum(r["verdict"] == "no_match" for r in forgery) / max(len(forgery), 1),
    }


def cedar_rows():
    _, _, test_ids = split_person_ids(seed=42)
    ids = [value for value in test_ids if value < 1000]
    dataset = SignatureDataset(person_ids=ids, augment=False)
    real, forge = defaultdict(list), defaultdict(list)
    for path, person in dataset.original_records: real[person].append(path)
    for path, person in dataset.forged_records: forge[person].append(path)
    for values in (*real.values(), *forge.values()): values.sort(key=str)
    paths = sorted({p for values in real.values() for p in values} | {p for values in forge.values() for p in values}, key=str)
    model = get_model().eval(); features = {}
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch = paths[start:start+32]
            values = model.forward_once(torch.cat([load_image(path) for path in batch])).cpu()
            for path, value in zip(batch, values): features[str(path)] = value
    output = {count: [] for count in COUNTS}
    for person in sorted(real):
        queries = [(path, "match") for path in real[person][10:]] + [(path, "no_match") for path in forge[person]]
        for query, expected in queries:
            all_distances = [float(F.pairwise_distance(features[str(ref)].unsqueeze(0), features[str(query)].unsqueeze(0))) for ref in real[person][:10]]
            for count in COUNTS:
                decision = decide_registered_person(all_distances[:count], THRESHOLD, T_DIFF, T_SAME)
                output[count].append({"expected": expected, "verdict": decision["verdict"]})
    return {str(count): summarize(rows) for count, rows in output.items()}


def utsig_rows():
    features = torch.load("outputs/transfer_head_20260711_235238/features.pt", map_location="cpu")
    head = PairHead().cpu().eval()
    head.load_state_dict(torch.load("models/utsig_v3_binary_head.pth", map_location="cpu"))
    _, _, test_ids = split_person_ids(seed=42)
    ids = [value for value in test_ids if value >= 2000]
    dataset = SignatureDataset(person_ids=ids, augment=False)
    real, forge = defaultdict(list), defaultdict(list)
    for path, person in dataset.original_records: real[person].append(path)
    for path, person in dataset.forged_records: forge[person].append(path)
    for values in (*real.values(), *forge.values()): values.sort(key=str)
    output = {count: [] for count in COUNTS}
    with torch.no_grad():
        for person in sorted(real):
            queries = [(path, "match") for path in real[person][10:]] + [(path, "no_match") for path in forge[person]]
            for query, expected in queries:
                probs = [torch.sigmoid(head(features[str(ref)].unsqueeze(0), features[str(query)].unsqueeze(0))).item() for ref in real[person][:10]]
                for count in COUNTS:
                    decision = decide_utsig_registered(probs[:count])
                    output[count].append({"expected": expected, "verdict": decision["verdict"]})
    return {str(count): summarize(rows) for count, rows in output.items()}


def archive_rows():
    people = load_archive_people()
    _, _, test_ids = split_people(people)
    binary = load_run("outputs/archive_transfer_v2_20260712_022322")
    gray = load_run("outputs/archive_transfer_v2_20260712_023115")
    output = {count: [] for count in COUNTS}
    for person in test_ids:
        queries = [(path, "match") for path in people[person]["T"][10:]] + [(path, "no_match") for path in people[person]["F"]]
        for query, expected in queries:
            probs = [archive_probability(ref, query, binary, gray) for ref in people[person]["T"][:10]]
            for count in COUNTS:
                decision = decide_archive_registered(probs[:count])
                output[count].append({"expected": expected, "verdict": decision["verdict"]})
    return {str(count): summarize(rows) for count, rows in output.items()}


def main():
    report = {"CEDAR_v1": cedar_rows(), "UTSig_v3": utsig_rows(), "Archive_v2": archive_rows()}
    Path("outputs/reference_count_report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)


if __name__ == "__main__": main()
