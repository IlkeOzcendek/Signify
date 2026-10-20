"""Safe short training experiment; never overwrites production checkpoints."""

import json
import random
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models.loss import ContrastiveLoss
from src.models.siamese_network import SiameseNetwork
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids


SEED = 42
EPOCHS = 3
MAX_POSITIVE_PAIRS_PER_PERSON = 40
MAX_VALIDATION_PAIRS = 2400
BATCH_SIZE = 32


def balanced_subset(dataset, limit):
    positive = [i for i, pair in enumerate(dataset.pairs) if pair[2] == 1]
    negative = [i for i, pair in enumerate(dataset.pairs) if pair[2] == 0]
    per_class = limit // 2
    return Subset(dataset, positive[:per_class] + negative[:per_class])


def collect(model, loader, device):
    labels, distances = [], []
    model.eval()
    with torch.no_grad():
        for image_1, image_2, batch_labels in loader:
            output_1, output_2 = model(image_1.to(device), image_2.to(device))
            values = F.pairwise_distance(output_1, output_2)
            labels.extend(int(value) for value in batch_labels.tolist())
            distances.extend(float(value) for value in values.cpu().tolist())
    return labels, distances


def choose_threshold(labels, distances):
    best = None
    for threshold in sorted(set(distances)):
        predictions = [int(value < threshold) for value in distances]
        tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
        tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
        fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
        fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
        accuracy = (tp + tn) / len(labels)
        far = fp / max(fp + tn, 1)
        frr = fn / max(fn + tp, 1)
        candidate = (accuracy, -far, -frr, threshold, far, frr)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    return {
        "accuracy": best[0],
        "threshold": best[3],
        "far": best[4],
        "frr": best[5],
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "mps" if torch.backends.mps.is_available()
        else "cuda" if torch.cuda.is_available()
        else "cpu"
    )
    run_dir = Path("outputs") / f"short_experiment_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    train_ids, validation_ids, test_ids = split_person_ids(seed=SEED)
    train_dataset = SignatureDataset(
        person_ids=train_ids,
        augment=True,
        seed=SEED,
        max_positive_pairs_per_person=MAX_POSITIVE_PAIRS_PER_PERSON,
    )
    validation_dataset = SignatureDataset(
        person_ids=validation_ids,
        augment=False,
        seed=SEED,
        max_positive_pairs_per_person=MAX_POSITIVE_PAIRS_PER_PERSON,
    )
    validation_sample = balanced_subset(validation_dataset, MAX_VALIDATION_PAIRS)

    train_loader = DataLoader(
        train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0, drop_last=True
    )
    validation_loader = DataLoader(
        validation_sample, batch_size=BATCH_SIZE, shuffle=False, num_workers=0
    )

    model = SiameseNetwork().to(device)
    criterion = ContrastiveLoss(margin=1.0)
    optimizer = torch.optim.Adam(model.parameters(), lr=5e-4, weight_decay=3e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1
    )

    history = []
    best_accuracy = -1.0
    print(
        f"device={device} train_pairs={len(train_dataset)} "
        f"validation_pairs={len(validation_sample)} output={run_dir}",
        flush=True,
    )

    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for batch_index, (image_1, image_2, labels) in enumerate(train_loader, start=1):
            optimizer.zero_grad()
            output_1, output_2 = model(image_1.to(device), image_2.to(device))
            loss = criterion(output_1, output_2, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            if batch_index % 100 == 0:
                print(f"epoch={epoch}/{EPOCHS} batch={batch_index}/{len(train_loader)}", flush=True)

        labels, distances = collect(model, validation_loader, device)
        result = choose_threshold(labels, distances)
        result.update({
            "epoch": epoch,
            "train_loss": total_loss / len(train_loader),
            "learning_rate": optimizer.param_groups[0]["lr"],
        })
        history.append(result)
        scheduler.step(result["accuracy"])

        if result["accuracy"] > best_accuracy:
            best_accuracy = result["accuracy"]
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            (run_dir / "best_metrics.json").write_text(
                json.dumps(result, indent=2), encoding="utf-8"
            )

        (run_dir / "history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )
        print(
            f"epoch={epoch} loss={result['train_loss']:.4f} "
            f"val_acc={result['accuracy']:.3%} FAR={result['far']:.3%} "
            f"FRR={result['frr']:.3%} threshold={result['threshold']:.4f}",
            flush=True,
        )

    metadata = {
        "device": str(device),
        "seed": SEED,
        "epochs": EPOCHS,
        "max_positive_pairs_per_person": MAX_POSITIVE_PAIRS_PER_PERSON,
        "train_people": train_ids,
        "validation_people": validation_ids,
        "held_out_test_people": test_ids,
        "note": "Test split was not used during training or checkpoint selection.",
    }
    (run_dir / "run_config.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(f"completed output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
