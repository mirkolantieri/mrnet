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
        self.metric = metric
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
        self.log_every = 50
        self.experiment_path = './experiments/'
        self.experiment_name = f'exp_{self.metric}'
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

    def train(self, writer: SummaryWriter, epoch: int) -> (np.ndarray, np.ndarray):
        """
           `train`: train the model

           args:
               writer: the writer for the tensorboard
               epoch: the current epoch
        """
        self.model.train()
        self.model = self.model.to(self.device)

        y_prediction = []
        y_label = []
        y_class_predictions = []
        losses = []

        for i, (image, label, weights) in enumerate(self.train_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            weights = weights.to(self.device)

            prediction = self.model(image.float())
            loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weights)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            loss_score = loss.item()
            losses.append(loss_score)

            score = torch.sigmoid(prediction)

            y_label.append(int(label[0]))
            y_prediction.append(score[0].item())
            y_class_predictions.append((score[0] > 0.5).float().item())

            try:
                accuracy = metrics.balanced_accuracy_score(y_label, y_prediction)
                if self.metric == 'accuracy':
                    auc = metrics.roc_auc_score(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score, epoch*len(self.train_loader)+i)
                    writer.add_scalar('Training AUC', auc, epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420: break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch+1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {np.round(accuracy, 4)}")
                    writer.add_scalar('Training AUC per epoch', auc, epoch)
                    writer.add_scalar('Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_accuracy = np.round(accuracy, 4)

                    return training_loss, training_accuracy
                if self.metric == 'auc':
                    auc = metrics.auc(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score, epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training AUC', auc, epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420: break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {np.round(accuracy, 4)}")
                    writer.add_scalar('Training AUC per epoch', auc, epoch)
                    writer.add_scalar('Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_auc = np.round(auc, 4)

                    return training_loss, training_auc

                if self.metric == 'wu':
                    wu = ut.weighted_utility(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score, epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training WU', wu, epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420: break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training WU {wu}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {accuracy}")
                    writer.add_scalar('Training WU per epoch', wu, epoch)
                    writer.add_scalar('Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_wu = np.round(wu, 4)

                    return training_loss, training_wu
            except:
                auc = 0.5
                accuracy = 0.5
        return

    def fit(self, writer: SummaryWriter, epoch: int) -> (np.ndarray, np.ndarray):
        """
            `train`: train the model
            args:
                writer: the writer for the tensorboard
                epoch: the current epoch
        """
        self.model.eval()
        self.model = self.to(self.device)

        y_prediction = []
        y_label = []
        y_class_predictions = []
        losses = []

        for i, (image, label, weights) in enumerate(self.test_loader):
            image = image.to(self.device)
            label = label.to(self.device)
            weights = weights.to(self.device)

            prediction = self.model.forward(image.float())
            loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weights)

            loss_score = loss.item()
            losses.append(loss_score)

            score = torch.sigmoid(prediction)

            y_label.append(int(label[0]))
            y_prediction.append(score[0].item())
            y_class_predictions.append((score[0] > 0.5).float().item())

            try:
                accuracy = metrics.balanced_accuracy_score(y_label, y_prediction)
                if self.metric == 'accuracy':
                    auc = metrics.roc_auc_score(y_label, y_prediction)
                    writer.add_scalar('Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation AUC', auc, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar('Validation AUC per epoch', auc, epoch)
                        writer.add_scalar('Validation accuracy per epoch', accuracy, epoch)

                        val_loss = np.round(np.mean(losses), 4)
                        val_accuracy_epoch = np.round(accuracy, 4)

                        val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_label, y_class_predictions)
                        val_accuracy, val_sensitivity, val_specificity = np.round(val_accuracy, 4), np.round(val_sensitivity, 4), np.round(val_specificity)

                        return val_loss, val_accuracy_epoch, val_accuracy, val_sensitivity, val_specificity
                if self.metric == 'auc':
                    auc = metrics.auc(y_label, y_prediction)
                    writer.add_scalar('Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation AUC', auc, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar('Validation AUC per epoch', auc, epoch)
                        writer.add_scalar('Validation accuracy per epoch', accuracy, epoch)

                        val_loss = np.round(np.mean(losses), 4)
                        val_accuracy_epoch = np.round(accuracy, 4)

                        val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_label,
                                                                                                             y_class_predictions)
                        val_accuracy, val_sensitivity, val_specificity = np.round(val_accuracy, 4), np.round(
                            val_sensitivity, 4), np.round(val_specificity)

                        return val_loss, val_accuracy_epoch, val_accuracy, val_sensitivity, val_specificity

                if self.metric == 'wu':
                    wu = ut.weighted_utility(y_label, y_prediction)
                    writer.add_scalar('Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation WU', wu, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation WU {np.round(wu, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar('Validation AUC per epoch', auc, epoch)
                        writer.add_scalar('Validation accuracy per epoch', accuracy, epoch)

                        val_loss = np.round(np.mean(losses), 4)
                        val_wu_epoch = np.round(wu, 4)

                        val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_label,
                                                                                                             y_class_predictions)
                        val_accuracy, val_sensitivity, val_specificity = np.round(val_accuracy, 4), np.round(
                            val_sensitivity, 4), np.round(val_specificity)

                        return val_loss, val_wu_epoch, val_accuracy, val_sensitivity, val_specificity
            except:
                auc = 0.5
                accuracy = 0.5


        return