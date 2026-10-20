"""Extract isolated grayscale MobileNet features for all UTSig images."""

from pathlib import Path

import torch

from src.models.transfer_siamese import MobileNetSignatureEncoder
from src.preprocessing.archive_grayscale_v2 import load_image
from src.preprocessing.signature_dataset import list_images


OUTPUT = Path("outputs/utsig_gray_features.pt")
BATCH_SIZE = 32


def main():
    paths = sorted(list_images(Path("dataset/UTSig")), key=str)
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    encoder = MobileNetSignatureEncoder(
        projection=False, freeze_backbone=True
    ).to(device).eval()
    features = {}
    with torch.no_grad():
        for start in range(0, len(paths), BATCH_SIZE):
            batch_paths = paths[start:start + BATCH_SIZE]
            batch = torch.stack([load_image(path) for path in batch_paths]).to(device)
            values = encoder.forward_once(batch).cpu()
            for path, value in zip(batch_paths, values):
                features[str(path)] = value
            if start % 640 == 0:
                print(f"features={min(start + BATCH_SIZE, len(paths))}/{len(paths)}", flush=True)
    torch.save(features, OUTPUT)
    print(f"output={OUTPUT} images={len(features)}", flush=True)


if __name__ == "__main__":
    main()
