"""Isolated skilled-forgery experiment with person-held-out evaluation."""

import hashlib
import json
import os
import random
import re
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_transfer_head import PairHead
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.archive_grayscale_v2 import load_image as load_gray
from verify import load_image as load_binary


ROOT = Path(os.environ.get("SIGNIFY_SKILLED_DATASET", "dataset/skilled_forgery"))
SEED = 42
MAX_PAIRS_PER_CLASS = 120


def load_people():
    if not ROOT.is_dir():
        raise SystemExit("Set SIGNIFY_SKILLED_DATASET to the skilled-forgery dataset directory.")
    people = defaultdict(lambda: {"real": {}, "forge": {}})
    for path in ROOT.rglob("*.png"):
        match = re.match(r"^(\d{3})(\d{2})(\d{3})$", path.stem)
        if not match:
            continue
        writer, _, target = match.groups()
        label = "forge" if path.parent.name == "forge" else "real"
        person = target if label == "forge" else writer
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        people[person][label].setdefault(digest, path)
    return {
        person: {
            label: sorted(values.values(), key=str)
            for label, values in kinds.items()
        }
        for person, kinds in people.items()
        if len(kinds["real"]) >= 5 and len(kinds["forge"]) >= 5
    }


def split(people):
    ids = sorted(people)
    random.Random(SEED).shuffle(ids)
    return ids[:30], ids[30:37], ids[37:]


def extract(paths, loader, device):
    encoder = MobileNetSignatureEncoder(
        projection=False, freeze_backbone=True
    ).to(device).eval()
    features = {}
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch_paths = paths[start:start + 32]
            images = (
                torch.cat([loader(path) for path in batch_paths])
                if loader is load_binary
                else torch.stack([loader(path) for path in batch_paths])
            ).to(device)
            values = encoder.forward_once(images).cpu()
            for path, value in zip(batch_paths, values):
                features[str(path)] = value
            print(f"features={min(start+32,len(paths))}/{len(paths)}", flush=True)
    return features


def make_pairs(people, ids, cap=True):
    rng = random.Random(SEED)
    result = []
    for person in ids:
        real, forge = people[person]["real"], people[person]["forge"]
        positive = [(a, b, 1.0) for a, b in combinations(real, 2)]
        negative = [(a, b, 0.0) for a in real for b in forge]
        rng.shuffle(positive)
        rng.shuffle(negative)
        if cap:
            positive = positive[:MAX_PAIRS_PER_CLASS]
            negative = negative[:MAX_PAIRS_PER_CLASS]
        result.extend(positive + negative)
    rng.shuffle(result)
    return result


def tensors(rows, features):
    return (
        torch.stack([features[str(a)] for a, _, _ in rows]),
        torch.stack([features[str(b)] for _, b, _ in rows]),
        torch.tensor([label for _, _, label in rows]),
    )


def probabilities(head, data):
    first, second, labels = data
    values = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(labels), 2048):
            values.append(torch.sigmoid(head(
                first[start:start + 2048], second[start:start + 2048]
            )))
    return torch.cat(values), labels


def metrics(probs, labels, threshold=0.5):
    prediction = probs >= threshold
    positive, negative = labels == 1, labels == 0
    return {
        "auc": roc_auc_score(labels.numpy(), probs.numpy()),
        "accuracy": float((prediction == positive).float().mean()),
        "far": float((prediction & negative).sum() / max(int(negative.sum()), 1)),
        "frr": float(((~prediction) & positive).sum() / max(int(positive.sum()), 1)),
        "pairs": len(labels),
    }


def train(train_data, validation_data, output):
    loader = DataLoader(TensorDataset(*train_data), batch_size=256, shuffle=True)
    head = PairHead().cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=3e-4)
    loss_fn = nn.BCEWithLogitsLoss()
    best_auc, stale, history = -1.0, 0, []
    for epoch in range(1, 31):
        head.train()
        total = 0.0
        for first, second, labels in loader:
            optimizer.zero_grad()
            loss = loss_fn(head(first, second), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        probs, labels = probabilities(head, validation_data)
        result = metrics(probs, labels)
        result.update({"epoch": epoch, "loss": total / len(loader)})
        history.append(result)
        print(f"{output.stem} epoch={epoch} validation={result}", flush=True)
        if result["auc"] > best_auc + 0.0005:
            best_auc, stale = result["auc"], 0
            torch.save(head.state_dict(), output)
        else:
            stale += 1
            if stale >= 6:
                break
    head.load_state_dict(torch.load(output, map_location="cpu"))
    return head, history


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    run = Path("outputs") / f"skilled_head_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)
    people = load_people()
    train_ids, validation_ids, test_ids = split(people)
    paths = sorted({p for kinds in people.values() for values in kinds.values() for p in values}, key=str)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    binary = extract(paths, load_binary, device)
    gray = extract(paths, load_gray, device)
    torch.save(binary, run / "binary_features.pt")
    torch.save(gray, run / "gray_features.pt")
    train_rows = make_pairs(people, train_ids)
    validation_rows = make_pairs(people, validation_ids, cap=False)
    test_rows = make_pairs(people, test_ids, cap=False)
    report = {"split": {"train": train_ids, "validation": validation_ids, "test": test_ids}}
    models = {}
    for name, features in (("binary", binary), ("gray", gray)):
        train_data = tensors(train_rows, features)
        validation_data = tensors(validation_rows, features)
        test_data = tensors(test_rows, features)
        head, history = train(train_data, validation_data, run / f"{name}_head.pth")
        val_probs, val_labels = probabilities(head, validation_data)
        test_probs, test_labels = probabilities(head, test_data)
        report[name] = {
            "validation": metrics(val_probs, val_labels),
            "test": metrics(test_probs, test_labels), "history": history,
        }
        models[name] = (val_probs, test_probs, val_labels, test_labels)
    best = None
    for step in range(11):
        weight = step / 10
        val_probs = weight * models["gray"][0] + (1-weight) * models["binary"][0]
        score = roc_auc_score(models["binary"][2].numpy(), val_probs.numpy())
        if best is None or score > best[0]:
            best = (score, weight)
    weight = best[1]
    val_probs = weight * models["gray"][0] + (1-weight) * models["binary"][0]
    test_probs = weight * models["gray"][1] + (1-weight) * models["binary"][1]
    report["fusion"] = {
        "gray_weight": weight,
        "validation": metrics(val_probs, models["binary"][2]),
        "test": metrics(test_probs, models["binary"][3]),
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"output={run}", flush=True)


if __name__ == "__main__":
    main()
