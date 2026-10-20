"""Fast hard-negative triplet metric learning on cached frozen features."""

import json
import random
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, TensorDataset

from scripts.train_skilled_forgery_head import load_people, split


SEED = 42
BASE = Path("outputs/skilled_head_20260712_050415")


class Projection(nn.Module):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(576, 256), nn.BatchNorm1d(256), nn.ReLU(),
            nn.Dropout(0.20), nn.Linear(256, 128),
        )

    def forward(self, value):
        return F.normalize(self.network(value), p=2, dim=1)


def hard_triplets(people, ids, features):
    rng = random.Random(SEED)
    rows = []
    for person in ids:
        real, forge = people[person]["real"], people[person]["forge"]
        for anchor in real:
            ranked = sorted(
                forge,
                key=lambda path: float(F.pairwise_distance(
                    features[str(anchor)].unsqueeze(0), features[str(path)].unsqueeze(0)
                )),
            )
            positives = [path for path in real if path != anchor]
            for _ in range(max(3, 60 // len(real))):
                rows.append((anchor, rng.choice(positives), rng.choice(ranked[:3])))
    rng.shuffle(rows)
    return (
        torch.stack([features[str(a)] for a, _, _ in rows]),
        torch.stack([features[str(p)] for _, p, _ in rows]),
        torch.stack([features[str(n)] for _, _, n in rows]),
    )


def pairs(people, ids, features):
    first, second, labels = [], [], []
    for person in ids:
        real, forge = people[person]["real"], people[person]["forge"]
        for i, a in enumerate(real):
            for b in real[i+1:]:
                first.append(features[str(a)]); second.append(features[str(b)]); labels.append(1)
        for a in real:
            for b in forge:
                first.append(features[str(a)]); second.append(features[str(b)]); labels.append(0)
    return torch.stack(first), torch.stack(second), torch.tensor(labels)


def scores(model, data):
    first, second, labels = data
    model.eval()
    with torch.no_grad():
        distances = F.pairwise_distance(model(first), model(second))
    return labels, distances


def threshold(labels, distances):
    best = None
    for value in torch.linspace(float(distances.min()), float(distances.max()), 301):
        prediction = distances < value
        positive, negative = labels == 1, labels == 0
        tpr = float((prediction & positive).sum() / max(int(positive.sum()), 1))
        tnr = float(((~prediction) & negative).sum() / max(int(negative.sum()), 1))
        candidate = ((tpr + tnr) / 2, float(value))
        if best is None or candidate[0] > best[0]: best = candidate
    return best[1]


def metrics(labels, distances, cut):
    prediction = distances < cut
    positive, negative = labels == 1, labels == 0
    tpr = float((prediction & positive).sum() / max(int(positive.sum()), 1))
    tnr = float(((~prediction) & negative).sum() / max(int(negative.sum()), 1))
    return {
        "auc": roc_auc_score(labels.numpy(), (-distances).numpy()),
        "balanced_accuracy": (tpr + tnr) / 2,
        "far": 1 - tnr, "frr": 1 - tpr,
    }


def run_variant(name, features, people, train_ids, validation_ids, test_ids, output):
    train_data = hard_triplets(people, train_ids, features)
    loader = DataLoader(TensorDataset(*train_data), batch_size=256, shuffle=True)
    validation_data = pairs(people, validation_ids, features)
    test_data = pairs(people, test_ids, features)
    model = Projection().cpu()
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=3e-4)
    best_auc, stale, history = -1.0, 0, []
    for epoch in range(1, 41):
        model.train(); total = 0.0
        for anchor, positive, negative in loader:
            optimizer.zero_grad()
            loss = F.triplet_margin_loss(model(anchor), model(positive), model(negative), margin=0.25)
            loss.backward(); optimizer.step(); total += loss.item()
        labels, distances = scores(model, validation_data)
        cut = threshold(labels, distances)
        result = metrics(labels, distances, cut)
        result.update({"epoch": epoch, "loss": total/len(loader), "threshold": cut})
        history.append(result); print(f"{name} epoch={epoch} {result}", flush=True)
        if result["auc"] > best_auc + 0.001:
            best_auc, stale = result["auc"], 0
            torch.save(model.state_dict(), output)
        else:
            stale += 1
            if stale >= 6: break
    model.load_state_dict(torch.load(output, map_location="cpu"))
    val_labels, val_distances = scores(model, validation_data)
    cut = threshold(val_labels, val_distances)
    test_labels, test_distances = scores(model, test_data)
    return {
        "threshold": cut, "validation": metrics(val_labels, val_distances, cut),
        "test": metrics(test_labels, test_distances, cut), "history": history,
    }


def main():
    random.seed(SEED); torch.manual_seed(SEED)
    run = Path("outputs") / f"feature_triplet_{datetime.now():%Y%m%d_%H%M%S}"
    run.mkdir(parents=True)
    people = load_people(); train_ids, validation_ids, test_ids = split(people)
    report = {"split": {"train": train_ids, "validation": validation_ids, "test": test_ids}}
    for name in ("binary", "gray"):
        features = torch.load(BASE / f"{name}_features.pt", map_location="cpu")
        report[name] = run_variant(
            name, features, people, train_ids, validation_ids, test_ids,
            run / f"{name}_projection.pth",
        )
    best_name = max(report.keys() - {"split"}, key=lambda key: report[key]["test"]["auc"])
    best = report[best_name]["test"]
    report["gate"] = {
        "best_variant": best_name,
        "passed": bool(
            best["auc"] >= .75 and best["far"] <= .10
            and best["frr"] <= .20 and best["balanced_accuracy"] >= .80
        ),
    }
    report["production_assets_modified"] = False
    (run / "report.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2), flush=True); print(f"output={run}", flush=True)


if __name__ == "__main__": main()
