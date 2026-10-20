"""Fast v2 gate: frozen MobileNet features + person-independent pair head."""

import json
import random
from datetime import datetime
from itertools import combinations
from pathlib import Path

import torch
import torch.nn as nn
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.evaluate_archive import ARCHIVE, parse_name
from scripts.finetune_archive_short import load_people, split_people
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.signature_dataset import SignatureDataset
from src.preprocessing.archive_grayscale_v2 import load_image as load_gray_image


SEED = 42
FEATURE_BATCH_SIZE = 32
HEAD_BATCH_SIZE = 512
MAX_EPOCHS = 40
PATIENCE = 5


class PairHead(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(576 * 2, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, feature_1, feature_2):
        pair = torch.cat(
            [torch.abs(feature_1 - feature_2), feature_1 * feature_2], dim=1
        )
        return self.network(pair).squeeze(1)


def extract_features(people, device):
    paths = sorted({
        path for kinds in people.values()
        for kind in ("T", "F") for path in kinds[kind]
    }, key=str)
    encoder = MobileNetSignatureEncoder(
        projection=False, freeze_backbone=True
    ).to(device).eval()
    features = {}
    with torch.no_grad():
        for start in range(0, len(paths), FEATURE_BATCH_SIZE):
            batch_paths = paths[start:start + FEATURE_BATCH_SIZE]
            images = torch.stack([load_gray_image(path) for path in batch_paths]).to(device)
            values = encoder.forward_once(images).cpu()
            for path, value in zip(batch_paths, values):
                features[str(path)] = value
            if start % (FEATURE_BATCH_SIZE * 10) == 0:
                print(f"features={min(start + FEATURE_BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)
    return features


def all_pairs(people, person_ids, balance=False):
    rng = random.Random(SEED)
    positives, negatives = [], []
    for person in person_ids:
        genuine = people[person]["T"]
        forged = people[person]["F"]
        positives.extend((a, b, 1.0) for a, b in combinations(genuine, 2))
        negatives.extend((a, b, 0.0) for a in genuine for b in forged)
    if balance and len(negatives) > len(positives):
        negatives = rng.sample(negatives, len(positives))
    pairs = positives + negatives
    rng.shuffle(pairs)
    return pairs


def tensors(pairs, features):
    return (
        torch.stack([features[str(a)] for a, _, _ in pairs]),
        torch.stack([features[str(b)] for _, b, _ in pairs]),
        torch.tensor([label for _, _, label in pairs], dtype=torch.float32),
    )


def probabilities(head, data, device):
    feature_1, feature_2, labels = data
    output = []
    head.eval()
    with torch.no_grad():
        for start in range(0, len(labels), 2048):
            logits = head(
                feature_1[start:start + 2048].to(device),
                feature_2[start:start + 2048].to(device),
            )
            output.append(torch.sigmoid(logits).cpu())
    return torch.cat(output), labels


def metrics(head, data, device, threshold=0.5):
    probs, labels = probabilities(head, data, device)
    predictions = (probs >= threshold).float()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "auc": roc_auc_score(labels.numpy(), probs.numpy()),
        "accuracy": (tp + tn) / len(labels),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
        "pairs": len(labels),
    }


def best_validation_threshold(head, data, device):
    probs, labels = probabilities(head, data, device)
    best = None
    for threshold in torch.linspace(0.05, 0.95, 181):
        predictions = (probs >= threshold).float()
        tp = int(((predictions == 1) & (labels == 1)).sum())
        tn = int(((predictions == 0) & (labels == 0)).sum())
        fp = int(((predictions == 1) & (labels == 0)).sum())
        fn = int(((predictions == 0) & (labels == 1)).sum())
        tpr = tp / max(tp + fn, 1)
        tnr = tn / max(tn + fp, 1)
        balanced_accuracy = (tpr + tnr) / 2
        candidate = (balanced_accuracy, -fp, float(threshold))
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return best[2]


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path("outputs") / f"archive_transfer_v2_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    people = load_people()
    train_ids, validation_ids, test_ids = split_people(people)
    features = extract_features(people, device)
    torch.save(features, run_dir / "features.pt")
    print("building pair tensors", flush=True)
    train_data = tensors(all_pairs(people, train_ids, balance=True), features)
    validation_data = tensors(all_pairs(people, validation_ids), features)
    test_data = tensors(all_pairs(people, test_ids), features)
    train_loader = DataLoader(
        TensorDataset(*train_data), batch_size=HEAD_BATCH_SIZE,
        shuffle=True, drop_last=True,
    )

    head = PairHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=5e-4, weight_decay=2e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_auc = -1.0
    stale = 0
    history = []
    print(
        f"device={device} split={train_ids}/{validation_ids}/{test_ids} "
        f"train_pairs={len(train_data[2])}", flush=True,
    )

    for epoch in range(1, MAX_EPOCHS + 1):
        head.train()
        loss_total = 0.0
        for feature_1, feature_2, labels in train_loader:
            optimizer.zero_grad()
            logits = head(feature_1.to(device), feature_2.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            loss_total += loss.item()
        validation = metrics(head, validation_data, device)
        validation.update({"epoch": epoch, "loss": loss_total / len(train_loader)})
        history.append(validation)
        print(f"epoch={epoch} validation={validation}", flush=True)
        if validation["auc"] > best_auc + 0.0005:
            best_auc = validation["auc"]
            stale = 0
            torch.save(head.state_dict(), run_dir / "best_head.pth")
        else:
            stale += 1
            if stale >= PATIENCE:
                print("early_stopping", flush=True)
                break

    head.load_state_dict(torch.load(run_dir / "best_head.pth", map_location=device))
    threshold = best_validation_threshold(head, validation_data, device)
    result = {
        "split": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "validation_threshold": threshold,
        "validation": metrics(head, validation_data, device, threshold),
        "test": metrics(head, test_data, device, threshold),
        "history": history,
    }
    (run_dir / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
