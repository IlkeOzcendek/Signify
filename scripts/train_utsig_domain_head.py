"""Train an isolated UTSig pair head on frozen features with person-held-out splits."""

import argparse
import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_transfer_head import PairHead
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids


SEED = 42
FEATURES = Path("outputs/transfer_head_20260711_235238/features.pt")
EPOCHS = 30
PATIENCE = 6
PAIRS_PER_CLASS_PER_PERSON = 120


def utsig_ids(ids):
    return [person_id for person_id in ids if person_id >= 2000]


def records(ids):
    dataset = SignatureDataset(person_ids=ids, augment=False, seed=SEED)
    genuine, forgery = {}, {}
    for path, person_id in dataset.original_records:
        genuine.setdefault(person_id, []).append(path)
    for path, person_id in dataset.forged_records:
        forgery.setdefault(person_id, []).append(path)
    for values in (*genuine.values(), *forgery.values()):
        values.sort(key=str)
    return genuine, forgery


def pairs(ids, balance=True):
    rng = random.Random(SEED)
    genuine, forgery = records(ids)
    positive, negative = [], []
    for person_id in sorted(genuine):
        originals = genuine[person_id]
        positives = [
            (originals[i], originals[j], 1.0)
            for i in range(len(originals))
            for j in range(i + 1, len(originals))
        ]
        negatives = [
            (original, forged, 0.0)
            for original in originals
            for forged in forgery.get(person_id, [])
        ]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        if balance:
            positive.extend(positives[:PAIRS_PER_CLASS_PER_PERSON])
            negative.extend(negatives[:PAIRS_PER_CLASS_PER_PERSON])
        else:
            positive.extend(positives)
            negative.extend(negatives)
    result = positive + negative
    rng.shuffle(result)
    return result


def tensors(pair_rows, features):
    return (
        torch.stack([features[str(a)] for a, _, _ in pair_rows]),
        torch.stack([features[str(b)] for _, b, _ in pair_rows]),
        torch.tensor([label for _, _, label in pair_rows]),
    )


def probabilities(head, data):
    first, second, labels = data
    values = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(labels), 2048):
            values.append(torch.sigmoid(head(
                first[start:start + 2048], second[start:start + 2048]
            )).cpu())
    return torch.cat(values), labels


def metrics(head, data, threshold=0.5):
    probs, labels = probabilities(head, data)
    predictions = probs >= threshold
    positive = labels == 1
    negative = labels == 0
    return {
        "auc": roc_auc_score(labels.numpy(), probs.numpy()),
        "accuracy": float((predictions == positive).float().mean()),
        "far": float((predictions & negative).sum() / max(int(negative.sum()), 1)),
        "frr": float(((~predictions) & positive).sum() / max(int(positive.sum()), 1)),
        "pairs": len(labels),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, default=FEATURES)
    parser.add_argument("--tag", default="binary")
    args = parser.parse_args()
    random.seed(SEED)
    torch.manual_seed(SEED)
    run_dir = Path("outputs") / f"utsig_domain_head_{args.tag}_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)
    train_ids, validation_ids, test_ids = split_person_ids(seed=SEED)
    train_ids, validation_ids, test_ids = map(utsig_ids, (train_ids, validation_ids, test_ids))
    features = torch.load(args.features, map_location="cpu")
    train_data = tensors(pairs(train_ids), features)
    validation_data = tensors(pairs(validation_ids, balance=False), features)
    test_data = tensors(pairs(test_ids, balance=False), features)
    loader = DataLoader(TensorDataset(*train_data), batch_size=256, shuffle=True)
    head = PairHead().cpu()
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=3e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_auc, stale, history = -1.0, 0, []
    for epoch in range(1, EPOCHS + 1):
        head.train()
        total = 0.0
        for first, second, labels in loader:
            optimizer.zero_grad()
            loss = criterion(head(first, second), labels)
            loss.backward()
            optimizer.step()
            total += loss.item()
        result = metrics(head, validation_data)
        result.update({"epoch": epoch, "loss": total / len(loader)})
        history.append(result)
        print(f"epoch={epoch} validation={result}", flush=True)
        if result["auc"] > best_auc + 0.0005:
            best_auc, stale = result["auc"], 0
            torch.save(head.state_dict(), run_dir / "best_head.pth")
        else:
            stale += 1
            if stale >= PATIENCE:
                break
    head.load_state_dict(torch.load(run_dir / "best_head.pth", map_location="cpu"))
    report = {
        "split": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "validation": metrics(head, validation_data),
        "test": metrics(head, test_data),
        "history": history,
        "features": str(args.features),
    }
    (run_dir / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
