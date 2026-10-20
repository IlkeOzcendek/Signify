import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from src.models.siamese_network import SiameseNetwork
from src.models.loss import ContrastiveLoss

from src.preprocessing.signature_dataset import (
    SignatureDataset,
    split_person_ids
)

if torch.backends.mps.is_available():
    device = torch.device("mps") # Uses the Apple Silicon GPU when it is available
elif torch.cuda.is_available():
    device = torch.device("cuda") # Uses the NVIDIA GPU when it is available
else:
    device = torch.device("cpu") # Uses the processor when there is no supported GPU

print("1 - Program started")
print("Device:", device)

MAX_POSITIVE_PAIRS_PER_PERSON = None # None = Uses every matching pair ------> more data helps the small model generalize instead of underfit

train_ids, val_ids, test_ids = split_person_ids() # IDs = Person numbers assigned to the train validation and test groups ------> every script uses the same split

train_dataset = SignatureDataset(
    person_ids = train_ids, # train_dataset = Signature pairs used for learning
    augment = True, # Augment = Creates small random variations ------> helps the model handle unseen signatures
    max_positive_pairs_per_person = MAX_POSITIVE_PAIRS_PER_PERSON
)

val_dataset = SignatureDataset(
    person_ids = val_ids, # val_dataset = Unseen pairs used to measure the model after each epoch
    augment = False, # Validation data stays unchanged for a fair comparison between epochs
    max_positive_pairs_per_person = MAX_POSITIVE_PAIRS_PER_PERSON
)

test_dataset = SignatureDataset(person_ids = test_ids, augment = False) # test_dataset = Held-out pairs used only for the final unbiased test

print("2 - Datasets loaded")
print("Train people:", train_ids)
print("Val people:", val_ids)
print("Test people:", test_ids)
print("Train count:", len(train_ids))
print("Val count:", len(val_ids))
print("Test count:", len(test_ids))
print("Train pairs:", len(train_dataset))
print("Val pairs:", len(val_dataset))
print("Test pairs:", len(test_dataset))

NUM_WORKERS = 0 # Worker = A separate process that loads data ------> zero avoids macOS spawn errors without a main guard
BATCH_SIZE = 32 # Batch = Pairs processed in one step ------> 32 fills MPS better and creates fewer steps per epoch

train_loader = DataLoader(
  train_dataset, # Loader = Divides the dataset into batches and sends them to the loop
  batch_size = BATCH_SIZE,
  shuffle = True, # Changes pair order every epoch ------> prevents learning from a fixed order
  num_workers = NUM_WORKERS,
  pin_memory = False, # Pin memory speeds up CUDA transfers but MPS does not support it
  drop_last = True # Drops an incomplete final batch ------> BatchNorm cannot learn from a batch with one pair
)

val_loader = DataLoader(
  val_dataset, # Holds validation batches used to compare epochs
  batch_size = BATCH_SIZE,
  shuffle = False, # Validation does not learn so the order can stay unchanged
  num_workers = NUM_WORKERS,
  pin_memory = False
)

test_loader = DataLoader(
  test_dataset, # Holds test batches used after training is complete
  batch_size = BATCH_SIZE,
  shuffle = False,
  num_workers = NUM_WORKERS,
  pin_memory = False
)

model = SiameseNetwork().to(device) # Siamese = Two identical branches that turn both signatures into comparable vectors

criterion = ContrastiveLoss() # Criterion = The rule used to calculate error ------> contrastive loss compares vector distances

METRIC_THRESHOLD = 0.85 # Temporary decision boundary for comparing epochs ------> evaluate.py finds and saves the final threshold

optimizer = torch.optim.Adam( # Optimizer = The algorithm that changes weights to reduce loss
  model.parameters(), # Parameters = Weights and biases that the optimizer updates
  lr = 0.0005, # Learning rate = Size of each weight update
  weight_decay = 3e-4 # L2 regularization = Penalizes large weights ------> helps reduce overfitting
)

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau( # Scheduler = Changes the learning rate while training
  optimizer,
  mode = "max", # Higher F1 is better
  factor = 0.5, # New learning rate = Current learning rate * 0.5
  patience = 1 # Waits one unimproved epoch before lowering the learning rate
)

num_epochs = 40 # Epoch = One complete pass over the training loader
best_val_f1 = -1.0 # Starts below every possible F1 score so the first result can become the best

patience = 5 # Stops training after five epochs without a better validation F1
epochs_without_improvement = 0 # Counts consecutive epochs without improvement

for epoch in range(num_epochs): # epoch = Index of the current full pass through the training loader
  model.train() # Enables training behavior for layers such as dropout and batch normalization
  total_train_loss = 0 # Stores the sum of every batch loss in the current epoch

  for batch_index, (img1, img2, label) in enumerate(train_loader): # batch_index = Position of the batch | img1 img2 = Pair tensors | label = Expected answer
    if batch_index % 100 == 0:
      print(f"Epoch {epoch + 1}/{num_epochs} | Batch {batch_index}/{len(train_loader)}")

    img1 = img1.to(device) # Moves the first signature batch to the selected device
    img2 = img2.to(device) # Moves the second signature batch to the same device
    label = label.to(device) # Label = The expected answer ------> 1 means matching and 0 means different

    output1, output2 = model(img1, img2) # output1 output2 = Embedding vectors of img1 and img2 ------> numerical signature representations

    loss = criterion(output1, output2, label) # loss = One number showing model error ------> same pair distance^2 | different pair max(margin - distance, 0)^2

    optimizer.zero_grad() # Clears old gradients because PyTorch adds gradients by default
    loss.backward() # Backpropagation = Calculates how much every weight contributed to the error
    optimizer.step() # Updates weights using the gradients and learning rate

    total_train_loss += loss.item() # item turns the one-value loss tensor into a Python number

  average_train_loss = total_train_loss / len(train_loader) # Represents the mean training error ------> sum of batch losses / number of train batches

  model.eval() # Switches layers such as dropout and batch normalization to evaluation mode
  total_val_loss = 0 # Stores the sum of every validation batch loss

  true_positive = 0 # Predicted matching and the pair really matches
  false_positive = 0 # Predicted matching but the pair is different
  true_negative = 0 # Predicted different and the pair really is different
  false_negative = 0 # Predicted different but the pair actually matches

  with torch.no_grad(): # Stops gradient tracking because validation measures the model without updating it
    for img1, img2, label in val_loader:
      img1 = img1.to(device)
      img2 = img2.to(device)
      label = label.to(device)

      output1, output2 = model(img1, img2) # Validation embeddings ------> only measured and never used to update weights

      loss = criterion(output1, output2, label) # Validation error for the current batch

      total_val_loss += loss.item()

      distances = F.pairwise_distance(output1, output2) # Euclidean distance ------> smaller means the signatures are more similar

      predictions = (distances < METRIC_THRESHOLD).float() # predictions = Model answers ------> distance < threshold becomes 1 otherwise 0

      true_positive += ((predictions == 1) & (label == 1)).sum().item()
      false_positive += ((predictions == 1) & (label == 0)).sum().item()
      true_negative += ((predictions == 0) & (label == 0)).sum().item()
      false_negative += ((predictions == 0) & (label == 1)).sum().item()

  average_val_loss = total_val_loss / len(val_loader) # Average validation loss = Sum of batch losses / Number of validation batches

  accuracy = (true_positive + true_negative) / (
    true_positive + true_negative + false_positive + false_negative
  ) # Accuracy = All correct predictions / All predictions

  precision = true_positive / (true_positive + false_positive + 1e-8) # Precision = Reliability of match predictions ------> correct matches / all predicted matches
  recall = true_positive / (true_positive + false_negative + 1e-8) # Recall = Ability to find real matches ------> found matches / all real matches
  f1 = 2 * precision * recall / (precision + recall + 1e-8) # F1 = One score representing precision and recall together ------> 1e-8 prevents division by zero

  scheduler.step(f1) # Sends validation F1 to the scheduler ------> lower the learning rate when improvement stops
  current_lr = optimizer.param_groups[0]["lr"] # Reads the learning rate currently used by Adam

  if f1 > best_val_f1:
    best_val_f1 = f1
    epochs_without_improvement = 0

    torch.save( # Saves the model when validation F1 reaches a new best score
      model.state_dict(), # state_dict = Every learned weight and buffer in the model
      "best_siamese_model.pth"
    )

    print("Best model saved (val F1)")

  else:
      epochs_without_improvement += 1
      print(f"No improvement for {epochs_without_improvement} epoch(s)")

  print(
    f"Epoch [{epoch + 1}/{num_epochs}] - "
    f"Train Loss: {average_train_loss:.4f} - "
    f"Val Loss: {average_val_loss:.4f} - "
    f"Val Acc: {accuracy:.4f} - "
    f"Val Prec: {precision:.4f} - "
    f"Val Recall: {recall:.4f} - "
    f"Val F1: {f1:.4f} - "
    f"LR: {current_lr:.6f}"
  )

  if epochs_without_improvement >= patience:
    print("Early stopping triggered") # Early stopping = Ends training when validation F1 no longer improves
    break

torch.save( # Saves the model from the final completed epoch
  model.state_dict(),
  "siamese_model.pth"
)

print("Best model is saved as best_siamese_model.pth")
print("Last epoch model is saved as siamese_model.pth")
