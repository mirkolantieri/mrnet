#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# trainer.py : script responsable for the training and evaluation of the AlexNet CNN model based on auc.
# The file contains the respective methods:
# *train_model* : the method trains a CNN for a given number of epoch, learning rate etc. using the trainer from the dataset
# *evaluate_model* : the method validates the implemented model
# *get_lr*: the method return the optimized learning rate during the model fitting
# *run*: the method runs the entire script when parsing the arguments via terminal


# Importing libraries

import argparse
import csv
import os
import random
import shutil
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from tensorboardX import SummaryWriter

from dataloader.loader import MRDataset
from helper import utils as ut
from models.model import AlexNet

torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    """
    Class `Trainer`: the class is used to train the model and optimize it
    by utilizing hyper-parameters tuning metrics (can be over accuracy, auroc or
    the weighted utility)
    """

    def __init__(self, metric: str = '', model: AlexNet = None, learning_rate: float = 1e-5, scheduler: str = 'step',
                 task: str = 'abnormal', plane: str = 'sagittal', epochs: int = 20) -> None:
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=1e-2)
        if scheduler == 'plateau':
            self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer, patience=5, factor=.3, threshold=1e-4, verbose=True
            )
        elif scheduler == 'step':
            self.scheduler = torch.optim.lr_scheduler.StepLR(
                self.optimizer, step_size=3, gamma=.5
            )
        self.train_set = MRDataset('./data/', task, plane, train=True)
        self.train_loader = torch.utils.data.DataLoader(
            self.train_set, batch_size=1, shuffle=True, num_workers=2, drop_last=True
        )
        self.test_set = MRDataset('./data/', task, plane, train=False)
        self.test_loader = torch.utils.data.DataLoader(
            self.test_set, batch_size=1, shuffle=True, num_workers=2, drop_last=True
        )
        self.epochs = epochs

        self.experiment_path = './experiments/'
        self.experiment_name = f'exp_{metric}'
        self.experiment_directory = os.path.join(self.experiment_path, self.experiment_name)
        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)
            os.makedirs(os.path.join(self.experiment_directory, 'models'))
            os.makedirs(os.path.join(self.experiment_directory, 'logs'))
            os.makedirs(os.path.join(self.experiment_directory, 'results'))
        self.log_root_path = (self.experiment_directory + f'/logs/{task}/{plane}') + datetime.now().\
            strftime("%Y%m%d""-%H%M%S") + "/"
        self.flush_history = 0

    @staticmethod
    def get_learning_rate(self):
        """
        `get_learning_rate`: returns the best learning rate from the parameters groups
        """
        for param in self.optimizer.param_groups:
            return param['lr']

    # TODO: train method and evaluation
    def train(self) -> (np.ndarray, np.ndarray):
        """
           `train_model`: train the model for a given number of epochs, learning rate, and an optimizer for the network model

           args:
               model: the model used for the training
               train_loader: the loader path of the training set
               epoch: the current epoch under performing of the train
               num_epochs: the total number of epochs to perform the training
               optimizer: the hyperparameter tuning configuration for the optimization of the model
               writer: the writer for the tensorboard
               current_lr: the current learning rate during the epoch
               device: the device to be used (gpu or cpu) for the model training
               log_every: the amount of logs to be processed
           """
        pass