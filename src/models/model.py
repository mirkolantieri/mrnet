#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# model.py : script responsible for the creation of the AlexNet CNN model
# The file contains the respective class: AlexNet(nn.Module) inherits from the nn.Module parent class
#


# Importing libraries
import torch
import torch.nn as nn
from torchvision import models

torch.multiprocessing.set_sharing_strategy("file_system")

# Reserved space for the implementation of the AlexNet based AI CNN (optimized already)


class AlexNet(nn.Module):
    """ class `AlexNet`: the implementation of the optimized AlexNet modern convolutional neural network
    to be used for the training and prediction. \\
    Weights initialization with "Imagenet"
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
