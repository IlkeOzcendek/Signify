from torch.utils.data import DataLoader
from src.preprocessing.signature_dataset import SignatureDataset

dataset = SignatureDataset()

loader = DataLoader(
  dataset, # Signiture pairs will be gotten from here
  batch_size = 16, # The data will be in groups of 16
  shuffle = True # Shuffle data at each epoch
)

for img1, img2, label in loader: # Get a batch from the loader
  print("Image 1 batch shape: ", img1.shape)
  print("Image 2 batch shape: ", img2.shape)

  print("Label batch shape: ", label.shape)

  print("Labels: ", label)

  break # loader can traverse through the whole dataset but just want to see the first batch
