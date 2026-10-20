"""Evaluate pretrained MobileNet features without fitting on validation/test."""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from scripts.benchmark_models import balanced_subset, best_threshold, metrics
from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids


def collect(model, loader, device):
    labels, distances = [], []
    with torch.no_grad():
        for image_1, image_2, batch_labels in loader:
            output_1, output_2 = model(image_1.to(device), image_2.to(device))
            values = F.pairwise_distance(output_1, output_2)
            labels.extend(int(value) for value in batch_labels.tolist())
            distances.extend(float(value) for value in values.cpu().tolist())
    return labels, distances


def main():
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    _, validation_ids, test_ids = split_person_ids()
    validation = balanced_subset(SignatureDataset(validation_ids, augment=False), 1200)
    test = balanced_subset(SignatureDataset(test_ids, augment=False), 1200)
    validation_loader = DataLoader(validation, batch_size=32, num_workers=0)
    test_loader = DataLoader(test, batch_size=32, num_workers=0)

    model = MobileNetSignatureEncoder(projection=False, freeze_backbone=True).to(device).eval()
    validation_labels, validation_distances = collect(model, validation_loader, device)
    validation_accuracy, threshold = best_threshold(validation_labels, validation_distances)
    test_labels, test_distances = collect(model, test_loader, device)
    result = metrics(test_labels, test_distances, threshold)
    print(
        f"device={device} threshold={threshold:.6f} val_acc={validation_accuracy:.3%} "
        f"test_acc={result['accuracy']:.3%} FAR={result['far']:.3%} FRR={result['frr']:.3%}"
    )


if __name__ == "__main__":
    main()
