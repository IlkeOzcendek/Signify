"""One guarded epoch on MobileNet's final block; isolated from production."""

import json
from datetime import datetime
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset

from scripts.finetune_archive_short import load_people, split_people, training_pairs
from scripts.train_archive_transfer_v2 import (
    PairHead, all_pairs, tensors, metrics, best_validation_threshold,
)
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.signature_dataset import SignatureDataset


SEED = 42
BATCH_SIZE = 16
HEAD_RUN = Path("outputs/archive_transfer_v2_20260712_022322")


class RawPairs(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs
        self.loader = SignatureDataset(person_ids=[], augment=True, seed=SEED)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path_1, path_2, label = self.pairs[index]
        return self.loader.load_image(path_1), self.loader.load_image(path_2), torch.tensor(label)


def encode(encoder, people, person_ids, device):
    paths = sorted({
        path for person in person_ids for kind in ("T", "F")
        for path in people[person][kind]
    }, key=str)
    loader = SignatureDataset(person_ids=[], augment=False, seed=SEED)
    features = {}
    encoder.eval()
    with torch.no_grad():
        for start in range(0, len(paths), 32):
            batch_paths = paths[start:start + 32]
            images = torch.stack([loader.load_image(path) for path in batch_paths]).to(device)
            values = encoder.forward_once(images).cpu()
            for path, value in zip(batch_paths, values):
                features[str(path)] = value
    return features


def main():
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    people = load_people()
    train_ids, validation_ids, test_ids = split_people(people)
    run_dir = Path("outputs") / f"archive_end2end_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    encoder = MobileNetSignatureEncoder(projection=False, freeze_backbone=True).to(device)
    # Open only the final two MobileNet feature blocks.
    for block in list(encoder.features.children())[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    head = PairHead().to(device)
    head.load_state_dict(torch.load(HEAD_RUN / "best_head.pth", map_location=device))

    pairs = training_pairs(people, train_ids)
    loader = DataLoader(RawPairs(pairs), batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    optimizer = torch.optim.AdamW([
        {"params": [p for p in encoder.parameters() if p.requires_grad], "lr": 1e-5},
        {"params": head.parameters(), "lr": 5e-5},
    ], weight_decay=1e-4)
    criterion = nn.BCEWithLogitsLoss()

    encoder.train(); head.train(); total_loss = 0.0
    for batch_index, (image_1, image_2, labels) in enumerate(loader, start=1):
        optimizer.zero_grad()
        feature_1 = encoder.forward_once(image_1.to(device))
        feature_2 = encoder.forward_once(image_2.to(device))
        logits = head(feature_1, feature_2)
        loss = criterion(logits, labels.float().to(device))
        loss.backward(); optimizer.step(); total_loss += loss.item()
        if batch_index % 25 == 0:
            print(f"batch={batch_index}/{len(loader)} loss={loss.item():.4f}", flush=True)

    validation_features = encode(encoder, people, validation_ids, device)
    test_features = encode(encoder, people, test_ids, device)
    validation_data = tensors(all_pairs(people, validation_ids), validation_features)
    test_data = tensors(all_pairs(people, test_ids), test_features)
    threshold = best_validation_threshold(head, validation_data, device)
    result = {
        "epoch_loss": total_loss / len(loader),
        "threshold": threshold,
        "validation": metrics(head, validation_data, device, threshold),
        "test": metrics(head, test_data, device, threshold),
    }
    torch.save(encoder.state_dict(), run_dir / "encoder.pth")
    torch.save(head.state_dict(), run_dir / "head.pth")
    (run_dir / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
