#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# model.py : script responsable for the creation of the AlexNet CNN model
# The file contains the respective class: AlexNet(nn.Module) inheritates from the nn.Module parent class
#


# Importing libraries
import torch
import torch.nn as nn
from torch.nn import functional as F
from torchvision import models

torch.multiprocessing.set_sharing_strategy('file_system')

# Reserved space for the implementation of the AlexNet based AI CNN (optimized already)


class AlexNet(nn.Module):
    """ class `AlexNet`: the implementation of the optimized AlexNet modern convolutional neural network
    to be used for the training and prediction. \\
    Weights initilization with "Imagenet"
    """

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=True)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.model.features(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x

# Reserved space for the implementation of the custom AlexNet based AI CNN (not optimized)


class AlexNetCustom(nn.Module):
    """ class `AlexNetCustom`: the implementation of the custom AlexNet modern convolutional neural network
    to be used for the training and prediction. \\
    Weights not initialized
    """

    def __init__(self):
        super().__init__()
        self.model = models.alexnet(pretrained=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(256, 1)

    def forward(self, x):
        x = torch.squeeze(x, dim=0)
        x = self.model.features(x)
        x = self.avg_pool(x).view(x.size(0), -1)
        x = torch.max(x, 0, keepdim=True)[0]
        x = self.classifier(x)
        return x
