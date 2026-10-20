"""Short, isolated archive fine-tune. Never overwrites production assets."""

import json
import random
from collections import defaultdict
from datetime import datetime
from itertools import combinations
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader, Dataset

from scripts.evaluate_archive import ARCHIVE, parse_name
from src.models.loss import ContrastiveLoss
from src.models.siamese_network_legacy import SiameseNetworkLegacy
from src.preprocessing.signature_dataset import SignatureDataset


SEED = 42
EPOCHS = 3
BATCH_SIZE = 32
PAIRS_PER_CLASS_PER_PERSON = 60
BASE_MODEL = Path("models/last_model_cedar_selected_utsig.pth")


def load_people():
    people = defaultdict(lambda: {"T": [], "F": []})
    for path in ARCHIVE.iterdir():
        parsed = parse_name(path)
        if parsed:
            person, kind, sample = parsed
            people[person][kind].append((sample, path))
    for kinds in people.values():
        for kind in kinds:
            kinds[kind] = [path for _, path in sorted(kinds[kind])]
    return {
        person: kinds for person, kinds in people.items()
        if len(kinds["T"]) >= 11 and kinds["F"]
    }


def split_people(people):
    ids = sorted(people)
    random.Random(SEED).shuffle(ids)
    return ids[:25], ids[25:31], ids[31:37]


def training_pairs(people, person_ids):
    rng = random.Random(SEED)
    pairs = []
    for person in person_ids:
        genuine = people[person]["T"]
        forged = people[person]["F"]
        positives = list(combinations(genuine, 2))
        negatives = [(a, b) for a in genuine for b in forged]
        rng.shuffle(positives)
        rng.shuffle(negatives)
        pairs.extend((a, b, 1.0) for a, b in positives[:PAIRS_PER_CLASS_PER_PERSON])
        pairs.extend((a, b, 0.0) for a, b in negatives[:PAIRS_PER_CLASS_PER_PERSON])
    rng.shuffle(pairs)
    return pairs


class PairDataset(Dataset):
    def __init__(self, pairs, augment):
        self.pairs = pairs
        # Reuse the exact unified application preprocessing and its versioned cache.
        self.loader = SignatureDataset(person_ids=[], augment=augment, seed=SEED)

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        path_1, path_2, label = self.pairs[index]
        return (
            self.loader.load_image(path_1),
            self.loader.load_image(path_2),
            torch.tensor(label, dtype=torch.float32),
        )


def load_model(device):
    model = SiameseNetworkLegacy().to(device)
    with torch.no_grad():
        dummy = torch.zeros(1, 1, 224, 224, device=device)
        model(dummy, dummy)
    model.load_state_dict(torch.load(BASE_MODEL, map_location=device))
    # Preserve the learned visual representation; adapt only the 128 -> 64 head.
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in model.fc[4].parameters():
        parameter.requires_grad = True
    return model


def encode(model, paths, image_loader, device):
    embeddings = {}
    model.eval()
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            images = torch.stack([image_loader.load_image(path) for path in batch_paths]).to(device)
            outputs = model.forward_once(images).cpu()
            for path, output in zip(batch_paths, outputs):
                embeddings[str(path)] = output
    return embeddings


def distance(embeddings, path_1, path_2):
    return float(F.pairwise_distance(
        embeddings[str(path_1)].unsqueeze(0),
        embeddings[str(path_2)].unsqueeze(0),
    ).item())


def evaluate_people(model, people, person_ids, device):
    paths = sorted({
        path for person in person_ids
        for kind in ("T", "F") for path in people[person][kind]
    }, key=str)
    image_loader = SignatureDataset(person_ids=[], augment=False, seed=SEED)
    embeddings = encode(model, paths, image_loader, device)
    labels, distances = [], []
    for person in person_ids:
        genuine = people[person]["T"]
        forged = people[person]["F"]
        for path_1, path_2 in combinations(genuine, 2):
            labels.append(1)
            distances.append(distance(embeddings, path_1, path_2))
        for path_1 in genuine:
            for path_2 in forged:
                labels.append(0)
                distances.append(distance(embeddings, path_1, path_2))

    auc = roc_auc_score(labels, [-value for value in distances])
    candidates = sorted(set(distances))
    best = None
    for threshold in candidates:
        predictions = [int(value < threshold) for value in distances]
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
        genuine_count = sum(labels)
        forgery_count = len(labels) - genuine_count
        far = sum(p == 1 and y == 0 for p, y in zip(predictions, labels)) / forgery_count
        frr = sum(p == 0 and y == 1 for p, y in zip(predictions, labels)) / genuine_count
        candidate = (accuracy, -far, threshold, far, frr)
        if best is None or candidate[:2] > best[:2]:
            best = candidate
    return {
        "auc": auc,
        "best_accuracy": best[0],
        "threshold": best[2],
        "far": best[3],
        "frr": best[4],
        "pairs": len(labels),
    }


def main():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    people = load_people()
    train_ids, validation_ids, test_ids = split_people(people)
    run_dir = Path("outputs") / f"archive_finetune_{datetime.now():%Y%m%d_%H%M%S}"
    run_dir.mkdir(parents=True)

    pairs = training_pairs(people, train_ids)
    loader = DataLoader(
        PairDataset(pairs, augment=True), batch_size=BATCH_SIZE,
        shuffle=True, num_workers=0, drop_last=True,
    )
    model = load_model(device)
    criterion = ContrastiveLoss(margin=1.2)
    optimizer = torch.optim.AdamW(
        [parameter for parameter in model.parameters() if parameter.requires_grad],
        lr=1e-4, weight_decay=1e-4,
    )

    baseline_validation = evaluate_people(model, people, validation_ids, device)
    baseline_test = evaluate_people(model, people, test_ids, device)
    print(f"split train={train_ids} validation={validation_ids} test={test_ids}", flush=True)
    print(f"baseline_validation={baseline_validation}", flush=True)
    print(f"baseline_test={baseline_test}", flush=True)

    best_auc = baseline_validation["auc"]
    history = []
    for epoch in range(1, EPOCHS + 1):
        model.train()
        total_loss = 0.0
        for image_1, image_2, labels in loader:
            optimizer.zero_grad()
            output_1, output_2 = model(image_1.to(device), image_2.to(device))
            loss = criterion(output_1, output_2, labels.to(device))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        validation = evaluate_people(model, people, validation_ids, device)
        validation["epoch"] = epoch
        validation["train_loss"] = total_loss / len(loader)
        history.append(validation)
        print(f"epoch={epoch} validation={validation}", flush=True)
        if validation["auc"] > best_auc:
            best_auc = validation["auc"]
            torch.save(model.state_dict(), run_dir / "best_model.pth")
            (run_dir / "best_validation.json").write_text(json.dumps(validation, indent=2))

    result = {
        "split": {"train": train_ids, "validation": validation_ids, "test": test_ids},
        "baseline_validation": baseline_validation,
        "baseline_test": baseline_test,
        "history": history,
        "improved": (run_dir / "best_model.pth").exists(),
    }
    if result["improved"]:
        best_model = load_model(device)
        best_model.load_state_dict(torch.load(run_dir / "best_model.pth", map_location=device))
        result["new_test"] = evaluate_people(best_model, people, test_ids, device)
    (run_dir / "report.json").write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2), flush=True)
    print(f"output={run_dir}", flush=True)


if __name__ == "__main__":
    main()
