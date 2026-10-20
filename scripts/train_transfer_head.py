"""Cache frozen MobileNet features, then train a lightweight pair classifier."""

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids


SEED = 42
FEATURE_BATCH_SIZE = 32
HEAD_BATCH_SIZE = 256
HEAD_EPOCHS = 20


class PairHead(nn.Module):
    def __init__(self, feature_size=576):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(feature_size * 2, 256),
            nn.ReLU(),
            nn.Dropout(0.25),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
        )

    def forward(self, feature_1, feature_2):
        pair = torch.cat([torch.abs(feature_1 - feature_2), feature_1 * feature_2], dim=1)
        return self.network(pair).squeeze(1)


def unique_paths(*datasets):
    paths = set()
    for dataset in datasets:
        for path, _ in dataset.original_records + dataset.forged_records:
            paths.add(path)
    return sorted(paths, key=str)


def extract_features(encoder, loader_dataset, paths, device):
    result = {}
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(paths), FEATURE_BATCH_SIZE):
            batch_paths = paths[start:start + FEATURE_BATCH_SIZE]
            images = torch.stack([loader_dataset.load_image(path) for path in batch_paths]).to(device)
            features = encoder.forward_once(images).cpu()
            for path, feature in zip(batch_paths, features):
                result[str(path)] = feature
            if start % (FEATURE_BATCH_SIZE * 20) == 0:
                print(f"features={min(start + FEATURE_BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)
    return result


def pair_tensors(dataset, features):
    feature_1, feature_2, labels = [], [], []
    for path_1, path_2, label in dataset.pairs:
        feature_1.append(features[str(path_1)])
        feature_2.append(features[str(path_2)])
        labels.append(float(label))
    return torch.stack(feature_1), torch.stack(feature_2), torch.tensor(labels)


def evaluate(head, tensors, device):
    feature_1, feature_2, labels = tensors
    head.eval()
    with torch.no_grad():
        probabilities = torch.sigmoid(
            head(feature_1.to(device), feature_2.to(device))
        ).cpu()
    predictions = (probabilities >= 0.5).float()
    tp = int(((predictions == 1) & (labels == 1)).sum())
    tn = int(((predictions == 0) & (labels == 0)).sum())
    fp = int(((predictions == 1) & (labels == 0)).sum())
    fn = int(((predictions == 0) & (labels == 1)).sum())
    return {
        "accuracy": (tp + tn) / len(labels),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
    }


def main():
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    run_dir = Path("outputs") / f"transfer_head_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    train_ids, validation_ids, test_ids = split_person_ids(seed=SEED)
    train = SignatureDataset(train_ids, augment=False, seed=SEED, max_positive_pairs_per_person=80)
    validation = SignatureDataset(validation_ids, augment=False, seed=SEED, max_positive_pairs_per_person=80)
    test = SignatureDataset(test_ids, augment=False, seed=SEED, max_positive_pairs_per_person=80)
    paths = unique_paths(train, validation, test)

    encoder = MobileNetSignatureEncoder(projection=False, freeze_backbone=True).to(device)
    features = extract_features(encoder, train, paths, device)
    torch.save(features, run_dir / "features.pt")
    print("building pair tensors", flush=True)
    train_tensors = pair_tensors(train, features)
    validation_tensors = pair_tensors(validation, features)
    test_tensors = pair_tensors(test, features)

    loader = DataLoader(TensorDataset(*train_tensors), batch_size=HEAD_BATCH_SIZE, shuffle=True)
    head = PairHead().to(device)
    optimizer = torch.optim.AdamW(head.parameters(), lr=1e-3, weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()
    best_accuracy = -1.0
    history = []

    for epoch in range(1, HEAD_EPOCHS + 1):
        head.train()
        total_loss = 0.0
        for feature_1, feature_2, labels in loader:
            optimizer.zero_grad()
            logits = head(feature_1.to(device), feature_2.to(device))
            loss = criterion(logits, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        result = evaluate(head, validation_tensors, device)
        result.update({"epoch": epoch, "loss": total_loss / len(loader)})
        history.append(result)
        print(
            f"epoch={epoch} loss={result['loss']:.4f} val_acc={result['accuracy']:.3%} "
            f"FAR={result['far']:.3%} FRR={result['frr']:.3%}", flush=True
        )
        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            torch.save(head.state_dict(), run_dir / "best_head.pth")
            (run_dir / "best_validation.json").write_text(json.dumps(result, indent=2))

    head.load_state_dict(torch.load(run_dir / "best_head.pth", map_location=device))
    test_result = evaluate(head, test_tensors, device)
    (run_dir / "test_metrics.json").write_text(json.dumps(test_result, indent=2))
    (run_dir / "history.json").write_text(json.dumps(history, indent=2))
    print(f"held_out_test={test_result} output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
