import torch

from torch.utils.data import DataLoader
from src.preprocessing.signature_dataset import (
    SignatureDataset,
    split_person_ids
)
from src.models.siamese_network import SiameseNetwork
from src.models.loss import ContrastiveLoss
from src.threshold import load_threshold


if torch.backends.mps.is_available():
  device = torch.device("mps") # Uses the Apple Silicon GPU when it is available
elif torch.cuda.is_available():
  device = torch.device("cuda") # Uses the NVIDIA GPU when it is available
else:
  device = torch.device("cpu") # Uses the processor when there is no supported GPU

train_ids, val_ids, test_ids = split_person_ids() # Split = Separates people into train validation and test groups ------> every script uses the same groups

test_dataset = SignatureDataset(person_ids = test_ids, augment = False) # Augment is off ------> test signatures must stay unchanged for a fair result

test_loader = DataLoader(
  test_dataset,
  batch_size = 16, # Tests 16 signature pairs in one step
  shuffle = False, # Keeps the test order unchanged
  num_workers = 0 # Worker = A separate process that loads data ------> zero avoids macOS spawn errors without a main guard
)

model = SiameseNetwork().to(device) # Siamese = Two identical branches that turn both signatures into comparable vectors

model.load_state_dict(
  torch.load("best_siamese_model.pth", map_location = device) # Loads the learned weights directly onto the selected device
)

model.eval() # Switches layers such as dropout and batch normalization to evaluation mode

criterion = ContrastiveLoss() # Contrastive loss pulls matching vectors closer and pushes different vectors apart

total_test_loss = 0 # Stores the sum of every batch loss
correct = 0 # Counts how many pair predictions match their labels
total = 0 # Counts every tested signature pair

threshold = load_threshold() # Threshold = The decision boundary found on validation data ------> a lower distance means the signatures match

with torch.no_grad(): # Stops gradient tracking because testing does not update the model
  for img1, img2, label in test_loader:
    img1 = img1.to(device) # Moves the first signature batch to the selected device
    img2 = img2.to(device) # Moves the second signature batch to the same device
    label = label.to(device) # Label = The expected answer ------> 1 means matching and 0 means different

    output1, output2 = model(img1, img2) # Output = An embedding vector that represents the signature with numbers

    loss = criterion(output1, output2, label) # Same pair ------> distance^2 | Different pair ------> max(margin - distance, 0)^2
    total_test_loss += loss.item() # item turns the one value tensor into a Python number

    distances = torch.nn.functional.pairwise_distance(output1, output2) # Euclidean distance ------> smaller means more similar
    predictions = (distances < threshold).float() # distance < threshold becomes 1 otherwise 0 ------> bool is converted to a number

    correct += (predictions == label).sum().item() # Compares answers and adds the number of True results
    total += label.size(0) # size(0) = Number of pairs in the current batch

if len(test_loader) == 0 or total == 0: # Prevents division by zero when the test set has no pairs
    raise ValueError("No test pairs were found.")

average_test_loss = total_test_loss / len(test_loader) # Average loss = Sum of batch losses / Number of batches
accuracy = correct / total # Accuracy = Correct predictions / All predictions ------> value stays between 0 and 1

print("Test people:", test_ids)
print(f"Test Loss: {average_test_loss:.4f}") # .4f shows four digits after the decimal point
print(f"Test Accuracy: {accuracy * 100:.2f}%") # Multiplies by 100 for percent and .2f keeps two decimal digits
