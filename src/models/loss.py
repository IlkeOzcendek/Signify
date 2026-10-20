import torch
import torch.nn as nn
import torch.nn.functional as F


class ContrastiveLoss(nn.Module):
  def __init__(self, margin = 1.0): # margin : What is the minimum distance threshold that different signatures should maintain?
    super().__init__()
    self.margin = margin

  def forward(self, output1, output2, label):
    euclidean_distance = F.pairwise_distance(output1, output2)

    loss = torch.mean(
       label * torch.pow(euclidean_distance, 2) +
       (1 - label) * torch.pow(
          torch.clamp(
            self.margin - euclidean_distance, min = 0.0
            ), 2
          ) # clamp rounds to 0 if it is negative and leave still if higher; in case if two different signatures already has enough distance among them therefore do not touch -> 0
    )
    return loss
