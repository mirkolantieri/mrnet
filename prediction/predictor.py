#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# predictor.py : script responsable for the prediction of the 0/1 (positive-negative) rates from the models
# The file contains the respective methods: extract_predictions
#

# Importing libraries
import csv
import os
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import tqdm
from sklearn import metrics
from sklearn.linear_model import LogisticRegression

from helper import utils as ut
from dataloader.loader import MRDataset
from models.model import AlexNet

torch.multiprocessing.set_sharing_strategy("file_system")

random.seed(42)
torch.manual_seed(42)
np.random.seed(42)
torch.cuda.manual_seed_all(42)


class Predictor:
    """Class `Predictor`: the class object is used
    to predict the cases for any abnormality. The cases to be predicted
    are the cases from the validation and the complex cases"""

    def __init__(self) -> None:
        pass

    def predict(self):
        pass

    def predict_complex(self):
        pass
