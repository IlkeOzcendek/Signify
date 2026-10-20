import torch
import torch.nn as nn


class ArchivePairHead(nn.Module): # It takes 2 signatures and creates the similarity record
    def __init__(self):
        super().__init__()

        self.network = nn.Sequential(
            nn.Linear(576 * 2, 256), # 2 signatures, because shortly in the forwarding process it will concatenate the properties of the two signatures to generate two sets of 576 bit information
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(0.30),
            nn.Linear(256, 64),
            nn.ReLU(),
            nn.Dropout(0.15),
            nn.Linear(64, 1),
        )

    def forward(self, feature_1, feature_2):
        pair = torch.cat( # Combine 2 signatures
            [torch.abs(feature_1 - feature_2), feature_1 * feature_2], dim = 1 # - for which are different, * for which are alike
        )
        return self.network(pair).squeeze(1)
