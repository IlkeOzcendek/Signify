from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import torch

image_path = "dataset/cedar/signatures/full_org/original_1_1.png"

# Original image
original = Image.open(image_path)

# Preprocessing
processed = original.convert("L")
processed = processed.resize((224, 224))

processed = np.array(processed)
processed = torch.tensor(processed, dtype=torch.float32)
processed = processed.unsqueeze(0)
processed = processed / 255.0

# Tensor -> image
processed_image = processed.squeeze(0).numpy()

plt.figure(figsize=(8,4))

plt.subplot(1,2,1)
plt.imshow(original, cmap="gray")
plt.title("Original Image")
plt.axis("off")

plt.subplot(1,2,2)
plt.imshow(processed_image, cmap="gray")
plt.title("Preprocessed (224×224)")
plt.axis("off")

plt.tight_layout()
plt.savefig("preprocessing_comparison.png", dpi=300)
plt.show()
