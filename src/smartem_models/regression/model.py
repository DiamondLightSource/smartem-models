import torch.nn as nn
from torchvision import models


class ResNetRegression(nn.Module):
    def __init__(self, freeze_base=True):
        super().__init__()
        # Load pretrained ResNet152
        self.resnet = models.resnet50(pretrained=True)
        # Freeze base parameters if needed
        if freeze_base:
            for param in self.resnet.parameters():
                param.requires_grad = False
        # Replace final layer to output a single regression value
        num_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Linear(num_features, 1)

    def forward(self, x):
        return self.resnet(x)
