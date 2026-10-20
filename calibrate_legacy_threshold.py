"""Find validation thresholds for the legacy model and save them to best_threshold.json.""" # Docstring = The official description stored for this module

# MODEL AND THRESHOLD MUST MATCH ------> legacy embeddings are not normalized so the current model threshold cannot be reused
# Run evaluate.py when returning to the current model ------> it rewrites best_threshold.json with thresholds for that model

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

from src.models.siamese_network_legacy import SiameseNetworkLegacy
from src.preprocessing.signature_dataset import SignatureDataset, split_person_ids
from src.threshold import save_thresholds
from verify import MODEL_PATH

if torch.backends.mps.is_available():
    device = torch.device("mps") # Uses the Apple Silicon GPU when it is available
elif torch.cuda.is_available():
    device = torch.device("cuda") # Uses the NVIDIA GPU when it is available
else:
    device = torch.device("cpu") # Uses the processor when there is no supported GPU

print("Device:", device)

_, val_ids, _ = split_person_ids() # _ ignores train and test IDs ------> only validation people are needed for calibration

val_dataset = SignatureDataset(person_ids = val_ids, augment = False) # Validation pairs stay unchanged so calibration is repeatable

val_loader = DataLoader(
    val_dataset, # Loader = Divides the validation dataset into batches
    batch_size = 16,
    shuffle = False,
    num_workers = 0 # Keeps the script compatible with macOS without a main guard
)

model = SiameseNetworkLegacy().to(device) # Creates the legacy architecture that matches the saved weights

with torch.no_grad(): # Stops gradient tracking because the dummy pass does not train the model
    model(
        torch.zeros(1, 1, 224, 224, device = device), # Dummy = Empty tensor with shape [batch channel height width]
        torch.zeros(1, 1, 224, 224, device = device)
    ) # Initializes the LazyLinear input size before loading its saved weights

model.load_state_dict(torch.load(MODEL_PATH, map_location = device)) # state_dict = Every learned weight and buffer in the model
model.eval() # Switches layers such as dropout and batch normalization to evaluation mode

print(f"Legacy model yüklendi: {MODEL_PATH}")


def collect_distances(loader):
    labels, distances = [], [] # labels = Expected answers | distances = Numerical differences between signature embeddings

    with torch.no_grad(): # Validation only measures the model and never updates its weights
        for img1, img2, batch_labels in loader:
            img1 = img1.to(device) # First signature batch
            img2 = img2.to(device) # Second signature batch

            o1, o2 = model(img1, img2) # o1 o2 = Embedding vectors that represent both signatures with numbers
            d = F.pairwise_distance(o1, o2) # Euclidean distance ------> smaller means the signatures are more similar

            labels.extend(batch_labels.numpy()) # extend adds every label instead of adding the whole batch as one item
            distances.extend(d.cpu().numpy()) # Moves distances to CPU and converts them into NumPy values

    return labels, distances


val_labels, val_distances = collect_distances(val_loader)

if not val_distances:
    raise ValueError("Validation çifti bulunamadı.")

dmax = max(val_distances) # dmax = Largest observed validation distance ------> defines the threshold search range

print(f"\nMesafe aralığı: 0 .. {dmax:.3f}  (legacy: normalize YOK, ölçek büyük)")

best_acc, best_threshold = 0.0, 0.0 # Stores the highest accuracy and the threshold that produced it
step = dmax / 60 if dmax > 0 else 0.05 # Tests 60 evenly spaced thresholds ------> fallback prevents a zero step

print("\nEşik taraması (VALIDATION)...\n")

t = step # t = Threshold currently being tested

while t <= dmax:
    preds = [1 if d < t else 0 for d in val_distances] # Distance below t becomes matching 1 otherwise different 0

    tp = sum(1 for p, l in zip(preds, val_labels) if p == 1 and l == 1) # True positive = Correctly predicted matching pair
    tn = sum(1 for p, l in zip(preds, val_labels) if p == 0 and l == 0) # True negative = Correctly predicted different pair

    acc = (tp + tn) / len(val_labels) # Accuracy = Correct predictions / All validation pairs

    if acc > best_acc:
        best_acc, best_threshold = acc, t # Keeps the threshold only when it produces a better accuracy

    t += step

CONF = 0.72 # Confidence requirement = At least 72% of labels in a region must support its decision
grid = [best_threshold * k / 20 for k in range(1, 41)] # Grid = 40 candidate boundaries from near zero to two times the best threshold

t_same, t_diff = 0.0, dmax # t_same = Confident match boundary | t_diff = Confident different boundary

for tt in grid:
    below = [l for l, d in zip(val_labels, val_distances) if d < tt] # Labels of pairs below the candidate match boundary

    if below and sum(below) / len(below) >= CONF:
        t_same = tt # Match ratio = Matching labels / All pairs below tt

for tt in grid:
    above = [l for l, d in zip(val_labels, val_distances) if d >= tt] # Labels of pairs above the candidate different boundary

    if above and sum(above) / len(above) <= (1 - CONF):
        t_diff = tt # Matching ratio must be at most 28% ------> at least 72% of this region is different
        break

save_thresholds(best_threshold, t_same, t_diff) # Saves the binary threshold and both confidence boundaries to JSON

print("==============================")
print(f"Best Threshold (val) : {best_threshold:.3f}")
print(f"Best Accuracy (val)  : {best_acc:.4f}")
print(f"'Kesin aynı'  < {t_same:.3f}")
print(f"'Kesin farklı' > {t_diff:.3f}")
print("best_threshold.json güncellendi (legacy model için).")
print("==============================")
