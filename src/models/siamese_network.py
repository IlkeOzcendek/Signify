import torch
import torch.nn as nn
import torch.nn.functional as F

class SiameseNetwork(nn.Module): # Siamese Network's aim is to produce the vectors that represents the two signatures
  def __init__(self): # Constructor, when "model = SiameseNetwork" is written, it works
    super().__init__() # Run the upper node's constructor (nn.Module.__init__(self) is the long way for it)

    self.cnn = nn.Sequential(
      nn.Conv2d(1, 32, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(32),
      nn.ReLU(),
      nn.MaxPool2d(2), # 112

      nn.Conv2d(32, 64, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(64),
      nn.ReLU(),
      nn.MaxPool2d(2), # 56

      nn.Conv2d(64, 128, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(128),
      nn.ReLU(),
      nn.MaxPool2d(2), # 28

      nn.Conv2d(128, 256, kernel_size = 3, padding = 1),
      nn.BatchNorm2d(256),
      nn.ReLU(),
      nn.MaxPool2d(2)  # 14
    )

    self.pool = nn.AdaptiveAvgPool2d((2, 2))

    self.fc = nn.Sequential(
      nn.Flatten(),
      nn.Linear(256 * 2 * 2, 128),
      nn.ReLU(),
      nn.Dropout(p = 0.5),
      nn.Linear(128, 128)
    )

  def forward_once(self, x):
    x = self.cnn(x)
    x = self.pool(x)
    x = self.fc(x)
    x = F.normalize(x, p = 2, dim = 1)

    return x

  def forward(self, img1, img2): # Both signatures are represented numerically
    output1 = self.forward_once(img1)
    output2 = self.forward_once(img2)

    return output1, output2
