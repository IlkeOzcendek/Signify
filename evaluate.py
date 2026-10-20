import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix
)
from src.models.siamese_network import SiameseNetwork
from src.preprocessing.signature_dataset import (
    SignatureDataset,
    split_person_ids,
)
from src.threshold import save_thresholds


if torch.backends.mps.is_available():
    device = torch.device("mps") # Uses the Apple Silicon GPU when it is available
elif torch.cuda.is_available():
    device = torch.device("cuda") # Uses the NVIDIA GPU when it is available
else:
    device = torch.device("cpu") # Uses the processor when there is no supported GPU

print("Device:", device)

train_ids, val_ids, test_ids = split_person_ids() # IDs = Person numbers assigned to train validation and test ------> every script uses the same split

print("Val people:", val_ids)
print("Test people:", test_ids)

val_dataset = SignatureDataset(person_ids = val_ids, augment = False) # Validation pairs choose thresholds without changing the model
test_dataset = SignatureDataset(person_ids = test_ids, augment = False) # Held-out test pairs measure the final result without influencing thresholds

val_loader = DataLoader(
    val_dataset, # Loader = Divides the validation dataset into batches
    batch_size = 16,
    shuffle = False,
    num_workers = 0 # Keeps the script compatible with macOS without a main guard
)

test_loader = DataLoader(
    test_dataset,
    batch_size = 16,
    shuffle = False,
    num_workers = 0
)


model = SiameseNetwork().to(device) # Siamese = Two identical branches that turn both signatures into comparable vectors

model.load_state_dict(
    torch.load("best_siamese_model.pth", map_location = device) # Loads saved tensors directly onto the selected device
) # state_dict = Every learned weight and buffer in the model

model.eval() # Switches layers such as dropout and batch normalization to evaluation mode

print("Best model loaded.")


def collect_distances(loader):
    labels = [] # Expected answers ------> 1 means matching and 0 means different
    distances = [] # Numerical differences between the two signature embeddings

    with torch.no_grad(): # Evaluation only measures the model and never updates its weights
        for img1, img2, batch_labels in loader:
            img1 = img1.to(device) # First signature batch
            img2 = img2.to(device) # Second signature batch

            output1, output2 = model(img1, img2) # Outputs = Embedding vectors that represent both signatures with numbers

            batch_distances = F.pairwise_distance(output1, output2) # Euclidean distance ------> smaller means the signatures are more similar

            labels.extend(batch_labels.numpy()) # extend adds each label instead of adding the entire batch as one item
            distances.extend(batch_distances.cpu().numpy()) # Moves distances to CPU and converts them into NumPy values

    return labels, distances


val_labels, val_distances = collect_distances(val_loader) # Validation determines the thresholds without seeing test results

if len(val_distances) == 0:
    raise ValueError("No validation pairs were found.")

best_accuracy = 0 # Highest validation accuracy found during the search
best_threshold = 0 # Distance boundary that produced the highest accuracy

# Accuracy is used instead of F1 ------> F1 ignores true negatives and can choose a loose threshold that accepts more forgeries
# FAR = False Acceptance Rate ------> Forged signatures incorrectly accepted as genuine

print("\nSearching best threshold on VALIDATION set...\n")

for threshold in [x / 20 for x in range(1, 40)]: # Normalized embedding distance ranges from 0 to 2 ------> tests boundaries every 0.05
    val_predictions = [
        1 if distance < threshold else 0 # Distance below the boundary becomes matching 1 otherwise different 0
        for distance in val_distances
    ]

    accuracy = accuracy_score(val_labels, val_predictions) # Accuracy = All correct predictions / All predictions
    precision = precision_score(val_labels, val_predictions, zero_division = 0) # Precision = Correct matches / All predicted matches
    recall = recall_score(val_labels, val_predictions, zero_division = 0) # Recall = Found matches / All real matches
    f1 = f1_score(val_labels, val_predictions, zero_division = 0) # F1 = Harmonic mean of precision and recall

    print(
        f"Threshold {threshold:.2f} | "
        f"Acc {accuracy:.4f} | "
        f"Prec {precision:.4f} | "
        f"Recall {recall:.4f} | "
        f"F1 {f1:.4f}"
    )

    if accuracy > best_accuracy:
        best_accuracy = accuracy # Keeps the new highest validation accuracy
        best_threshold = threshold # Keeps the boundary that produced it


CONF = 0.72 # Confidence requirement = At least 72% of labels in a region must support its decision
grid = [x / 40 for x in range(1, 80)] # Grid = Candidate confidence boundaries every 0.025 between 0 and 2

t_same = 0.0 # Largest boundary where at least 72% of lower distances are genuine
t_diff = 2.0 # Smallest boundary where at least 72% of higher distances are forged

for t in grid:
    below = [lb for lb, d in zip(val_labels, val_distances) if d < t] # Labels below the candidate genuine boundary

    if below and (sum(below) / len(below)) >= CONF:
        t_same = t # Genuine ratio = Matching labels / All pairs below t

for t in grid:
    above = [lb for lb, d in zip(val_labels, val_distances) if d >= t] # Labels above the candidate forgery boundary

    if above and (sum(above) / len(above)) <= (1 - CONF):
        t_diff = t # Matching ratio must be at most 28% ------> at least 72% of this region is forged
        break

uncertain = sum(1 for d in val_distances if t_same <= d < t_diff) # uncertain = Pairs whose distance stays between both confidence boundaries
uncertain_pct = 100 * uncertain / len(val_distances) # Uncertain percentage = Uncertain pairs / All validation pairs * 100

save_thresholds(best_threshold, t_same, t_diff) # Saves the binary threshold and confidence boundaries for test.py verify.py and the app

print("\n==============================")
print(f"Best Threshold (val)  : {best_threshold:.2f}  (binary decision)")
print(f"Best Accuracy (val)   : {best_accuracy:.4f}")
print(f"'Certain match' boundary    : distance < {t_same:.2f}")
print(f"'Certain no match' boundary : distance > {t_diff:.2f}")
print(f"'Uncertain' band             : {uncertain_pct:.1f}%  <-- this should be low")
print("Thresholds saved to best_threshold.json")
print("==============================")


test_labels, test_distances = collect_distances(test_loader) # Held-out = Data never used for training or threshold selection

if len(test_distances) == 0:
    raise ValueError("No test pairs were found.")

test_predictions = [
    1 if distance < best_threshold else 0 # Applies the validation threshold without adjusting it to test results
    for distance in test_distances
]

accuracy = accuracy_score(test_labels, test_predictions)
precision = precision_score(test_labels, test_predictions, zero_division = 0)
recall = recall_score(test_labels, test_predictions, zero_division = 0)
f1 = f1_score(test_labels, test_predictions, zero_division = 0)
cm = confusion_matrix(test_labels, test_predictions) # Matrix = [[true negative false positive] [false negative true positive]]

print("\n========== TEST RESULTS (held-out) ==========\n")
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

print("\nConfusion Matrix:\n")
print(cm)
