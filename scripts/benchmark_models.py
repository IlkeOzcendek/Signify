"""Compare every compatible checkpoint on the same validation/test split.

Threshold selection uses validation only; reported metrics use held-out test data.
The script is read-only and does not overwrite the application's threshold file.
"""

from pathlib import Path
import sys

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset

from src.models.siamese_network import SiameseNetwork
from src.models.siamese_network_legacy import SiameseNetworkLegacy
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids


DEVICE = torch.device(
    "mps" if torch.backends.mps.is_available()
    else "cuda" if torch.cuda.is_available()
    else "cpu"
)
MAX_PAIRS_PER_SPLIT = 1200


def balanced_subset(dataset, limit):
    """Deterministically keep up to half genuine and half forgery pairs."""
    positive, negative = [], []
    for index, (_, _, label) in enumerate(dataset.pairs):
        (positive if label == 1 else negative).append(index)
    per_class = limit // 2
    return Subset(dataset, positive[:per_class] + negative[:per_class])


def architecture_for(checkpoint):
    state = torch.load(checkpoint, map_location="cpu")
    first_weight = state.get("cnn.0.weight")
    if first_weight is not None and first_weight.shape[0] == 16:
        return SiameseNetworkLegacy
    if first_weight is not None and first_weight.shape[0] == 32:
        return SiameseNetwork
    raise ValueError("unknown checkpoint architecture")


def load_model(checkpoint):
    model_class = architecture_for(checkpoint)
    model = model_class().to(DEVICE)
    if model_class is SiameseNetworkLegacy:
        with torch.no_grad():
            dummy = torch.zeros(1, 1, 224, 224, device=DEVICE)
            model(dummy, dummy)
    model.load_state_dict(torch.load(checkpoint, map_location=DEVICE))
    model.eval()
    return model, model_class.__name__


def distances(model, loader):
    labels, values = [], []
    with torch.no_grad():
        for image_1, image_2, batch_labels in loader:
            output_1, output_2 = model(image_1.to(DEVICE), image_2.to(DEVICE))
            batch_distances = F.pairwise_distance(output_1, output_2)
            labels.extend(int(value) for value in batch_labels.tolist())
            values.extend(float(value) for value in batch_distances.cpu().tolist())
    return labels, values


def best_threshold(labels, values):
    candidates = sorted(set(values))
    best = (-1.0, 0.0)
    for threshold in candidates:
        predictions = [int(value < threshold) for value in values]
        accuracy = sum(p == y for p, y in zip(predictions, labels)) / len(labels)
        if accuracy > best[0]:
            best = accuracy, threshold
    return best


def metrics(labels, values, threshold):
    predictions = [int(value < threshold) for value in values]
    tp = sum(p == 1 and y == 1 for p, y in zip(predictions, labels))
    tn = sum(p == 0 and y == 0 for p, y in zip(predictions, labels))
    fp = sum(p == 1 and y == 0 for p, y in zip(predictions, labels))
    fn = sum(p == 0 and y == 1 for p, y in zip(predictions, labels))
    return {
        "accuracy": (tp + tn) / len(labels),
        "far": fp / max(fp + tn, 1),
        "frr": fn / max(fn + tp, 1),
    }


def abstention_bands(labels, values, target_accuracy=0.75):
    """Choose validation bands with maximum coverage at target decided accuracy."""
    ordered = sorted(values)
    grid = [ordered[round(i * (len(ordered) - 1) / 50)] for i in range(51)]
    best = None
    for same_threshold in grid:
        for diff_threshold in grid:
            if same_threshold > diff_threshold:
                continue
            decided = [
                (int(value < same_threshold), label)
                for label, value in zip(labels, values)
                if value < same_threshold or value >= diff_threshold
            ]
            if not decided:
                continue
            accuracy = sum(prediction == label for prediction, label in decided) / len(decided)
            coverage = len(decided) / len(labels)
            if accuracy >= target_accuracy and (best is None or coverage > best[0]):
                best = coverage, accuracy, same_threshold, diff_threshold
    return best


def abstention_metrics(labels, values, same_threshold, diff_threshold):
    decided = [
        (int(value < same_threshold), label)
        for label, value in zip(labels, values)
        if value < same_threshold or value >= diff_threshold
    ]
    accuracy = sum(prediction == label for prediction, label in decided) / max(len(decided), 1)
    return accuracy, len(decided) / len(labels)


def main():
    _, validation_ids, test_ids = split_person_ids()
    validation = SignatureDataset(person_ids=validation_ids, augment=False)
    test = SignatureDataset(person_ids=test_ids, augment=False)
    validation_sample = balanced_subset(validation, MAX_PAIRS_PER_SPLIT)
    test_sample = balanced_subset(test, MAX_PAIRS_PER_SPLIT)
    validation_loader = DataLoader(validation_sample, batch_size=32, num_workers=0)
    test_loader = DataLoader(test_sample, batch_size=32, num_workers=0)

    checkpoints = sorted(set(Path(".").glob("*.pth")) | set(Path("models").glob("*.pth")))
    if len(sys.argv) > 1:
        requested = Path(sys.argv[1])
        checkpoints = [requested]
    print(
        f"device={DEVICE} validation_pairs={len(validation_sample)}/{len(validation)} "
        f"test_pairs={len(test_sample)}/{len(test)}",
        flush=True,
    )
    print("checkpoint\tarchitecture\tthreshold\tval_acc\ttest_acc\tFAR\tFRR\tdecided_acc\tcoverage")
    for checkpoint in checkpoints:
        try:
            model, architecture = load_model(checkpoint)
            validation_labels, validation_values = distances(model, validation_loader)
            validation_accuracy, threshold = best_threshold(validation_labels, validation_values)
            test_labels, test_values = distances(model, test_loader)
            result = metrics(test_labels, test_values, threshold)
            bands = abstention_bands(validation_labels, validation_values)
            if bands is None:
                decided_accuracy, coverage = 0.0, 0.0
                same_threshold, diff_threshold = 0.0, 0.0
            else:
                _, _, same_threshold, diff_threshold = bands
                decided_accuracy, coverage = abstention_metrics(
                    test_labels, test_values, same_threshold, diff_threshold
                )
            print(
                f"{checkpoint}\t{architecture}\t{threshold:.6f}\t"
                f"{validation_accuracy:.3%}\t{result['accuracy']:.3%}\t"
                f"{result['far']:.3%}\t{result['frr']:.3%}\t"
                f"{decided_accuracy:.3%}\t{coverage:.3%}\t"
                f"bands=({same_threshold:.6f},{diff_threshold:.6f})",
                flush=True,
            )
        except Exception as error:
            print(f"{checkpoint}\tERROR\t{error}")


if __name__ == "__main__":
    main()
