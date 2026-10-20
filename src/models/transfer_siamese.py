import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import MobileNet_V3_Small_Weights, mobilenet_v3_small

class MobileNetSignatureEncoder(nn.Module):
    """ImageNet - pretrained signature encoder with optional trainable projection.""" # The docstring is stored by Python as the official description of the class

    def __init__(self, projection = True, freeze_backbone = True):
        super().__init__()
        backbone = mobilenet_v3_small(weights=MobileNet_V3_Small_Weights.DEFAULT) # It brings back the MobileNet that it trained before

        self.features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d(1) # By averaging all the pixels in each channel the spatial dimension becomes 1×1
        self.projection = (
            nn.Sequential(
                nn.Linear(576, 256),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(256, 128),
            )
            if projection else nn.Identity() # Do nothing in that case, v2 and v3 uses that
        )
        self.register_buffer("mean", torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)) # Buffer uses the model (cupa, mps)
        self.register_buffer("std", torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))

        if freeze_backbone:
            for parameter in self.features.parameters(): # MobileNet's learning parameters
                parameter.requires_grad = False

    def forward_once(self, image):
        image = image.repeat(1, 3, 1, 1) # [batch, 3, height, width] ------> 1 -> Repeat batch size, 3 -> Repeat channel size three times, 1 -> Repeat height, 1 -> Repeat width :::: repeat: It brings the grey signature to the 3 channels that MobileNet has been waiting for, RGB
        image = (image - self.mean) / self.std # It brings the pixel values ​​to the scale that MobileNet is accustomed to in education

        embedding = self.pool(self.features(image)).flatten(1)
        embedding = self.projection(embedding)

        return F.normalize(embedding, p = 2, dim = 1)

    def forward(self, image_1, image_2):
        return self.forward_once(image_1), self.forward_once(image_2)
