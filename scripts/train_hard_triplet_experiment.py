"""Short isolated hard-negative triplet fine-tune; never touches production assets."""

import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset
from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

from scripts.train_skilled_forgery_head import load_people, split
from src.preprocessing.archive_grayscale_v2 import load_image as load_gray
from verify import load_image as load_binary


SEED = 42
SHORT_EPOCHS = 6
LONG_EPOCHS = 30
TRIPLETS_PER_PERSON = 40
MARGIN = 0.25


class TripletEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT)
        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.projection = nn.Sequential(
            nn.Linear(576, 256), nn.ReLU(), nn.Dropout(0.2), nn.Linear(256, 128)
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
        for parameter in self.features.parameters():
            parameter.requires_grad = False
        # Fine-tune only the final feature block; the rest remains a stable prior.
        for parameter in self.features[-1].parameters():
            parameter.requires_grad = True

    def forward(self, image):
        image = image.repeat(1, 3, 1, 1)
        image = (image - self.mean) / self.std
        value = self.pool(self.features(image)).flatten(1)
        return F.normalize(self.projection(value), p=2, dim=1)


def load_view(path, gray=False):
    return load_gray(path) if gray else load_binary(path).squeeze(0)


class HardTriplets(Dataset):
    def __init__(self, people, ids, frozen_features):
        self.rows = []
        rng = random.Random(SEED)
        for person in ids:
            real, forge = people[person]["real"], people[person]["forge"]
            candidates = []
            for anchor in real:
                anchor_feature = frozen_features[str(anchor)]
                ranked = sorted(
                    forge,
                    key=lambda path: float(F.pairwise_distance(
                        anchor_feature.unsqueeze(0),
                        frozen_features[str(path)].unsqueeze(0),
                    )),
                )
                positives = [path for path in real if path != anchor]
                for _ in range(max(1, TRIPLETS_PER_PERSON // len(real))):
                    candidates.append((anchor, rng.choice(positives), rng.choice(ranked[:3])))
            rng.shuffle(candidates)
            self.rows.extend(candidates[:TRIPLETS_PER_PERSON])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        anchor, positive, negative = self.rows[index]
        # Deterministic alternating views retains thin strokes while teaching
        # tolerance to binary/grayscale preprocessing.
        gray = index % 2 == 1
        return load_view(anchor, gray), load_view(positive, gray), load_view(negative, gray)


def embed(model, paths, device):
    result = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch_paths = paths[start:start + 32]
            # Evaluation uses production binary preprocessing.
            images = torch.stack([load_view(path, False) for path in batch_paths]).to(device)
            values = model(images).cpu()
            for path, value in zip(batch_paths, values):
                result[str(path)] = value
    return result


def pair_rows(people, ids):
    rows = []
    for person in ids:
        real, forge = people[person]["real"], people[person]["forge"]
        rows.extend((a, b, 1) for i, a in enumerate(real) for b in real[i+1:])
        rows.extend((a, b, 0) for a in real for b in forge)
    return rows


def evaluate(model, people, ids, device):
    rows = pair_rows(people, ids)
    paths = sorted({path for a, b, _ in rows for path in (a, b)}, key=str)
    features = embed(model, paths, device)
    labels = torch.tensor([label for _, _, label in rows])
    distances = torch.tensor([
        float(F.pairwise_distance(features[str(a)].unsqueeze(0), features[str(b)].unsqueeze(0)))
        for a, b, _ in rows
    ])
    return labels, distances


def best_threshold(labels, distances):
    best = None
    for threshold in torch.linspace(float(distances.min()), float(distances.max()), 201):
        predictions = distances < threshold
        positive, negative = labels == 1, labels == 0
        tpr = float((predictions & positive).sum() / max(int(positive.sum()), 1))
        tnr = float(((~predictions) & negative).sum() / max(int(negative.sum()), 1))
        candidate = ((tpr + tnr) / 2, float(threshold))
        if best is None or candidate[0] > best[0]:
            best = candidate
    return best[1]


def metrics(labels, distances, threshold):
    prediction = distances < threshold
    positive, negative = labels == 1, labels == 0
    return {
        "auc": roc_auc_score(labels.numpy(), (-distances).numpy()),
        "balanced_accuracy": (
            float((prediction & positive).sum() / max(int(positive.sum()), 1))
            + float(((~prediction) & negative).sum() / max(int(negative.sum()), 1))
        ) / 2,
        "far": float((prediction & negative).sum() / max(int(negative.sum()), 1)),
        "frr": float(((~prediction) & positive).sum() / max(int(positive.sum()), 1)),
    }


def main():
    random.seed(SEED)
    torch.manual_seed(SEED)
    run = Path("outputs") / f"triplet_experiment_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)
    people = load_people()
    train_ids, validation_ids, test_ids = split(people)
    baseline_run = Path("outputs/skilled_head_20260712_050415")
    frozen = torch.load(baseline_run / "binary_features.pt", map_location="cpu")
    dataset = HardTriplets(people, train_ids, frozen)
    loader = DataLoader(dataset, batch_size=24, shuffle=True, num_workers=0, drop_last=True)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = TripletEncoder().to(device)
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=2e-4, weight_decay=3e-4
    )
    history, best_auc, stale = [], -1.0, 0
    max_epochs = SHORT_EPOCHS
    for epoch in range(1, max_epochs + 1):
        model.train()
        total = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            batch_size = anchor.shape[0]
            combined = torch.cat([anchor, positive, negative], dim=0).to(device)
            a, p, n = model(combined).split(batch_size)
            loss = F.triplet_margin_loss(a, p, n, margin=MARGIN)
            loss.backward()
            optimizer.step()
            total += loss.item()
        val_labels, val_distances = evaluate(model, people, validation_ids, device)
        threshold = best_threshold(val_labels, val_distances)
        result = metrics(val_labels, val_distances, threshold)
        result.update({"epoch": epoch, "loss": total / len(loader), "threshold": threshold})
        history.append(result)
        print(f"epoch={epoch} validation={result}", flush=True)
        if result["auc"] > best_auc + 0.002:
            best_auc, stale = result["auc"], 0
            torch.save(model.state_dict(), run / "best_short.pth")
        else:
            stale += 1
            if stale >= 3:
                break
    model.load_state_dict(torch.load(run / "best_short.pth", map_location=device))
    val_labels, val_distances = evaluate(model, people, validation_ids, device)
    threshold = best_threshold(val_labels, val_distances)
    test_labels, test_distances = evaluate(model, people, test_ids, device)
    validation = metrics(val_labels, val_distances, threshold)
    test = metrics(test_labels, test_distances, threshold)
    passed = (
        test["auc"] >= 0.75 and test["far"] <= 0.10
        and test["frr"] <= 0.20 and test["balanced_accuracy"] >= 0.80
    )
    report = {
        "split": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "short": {"threshold": threshold, "validation": validation, "test": test,
                  "history": history, "passed_long_gate": passed},
        "production_assets_modified": False,
    }
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True)
    print(f"output={run}", flush=True)


if __name__ == "__main__":
    main()
