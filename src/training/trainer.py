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

import csv
import os
import random
import time
from datetime import datetime

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim
from dataloader.loader import MRDataset
from helper import utils as ut
from models.model import AlexNet
from numpy.random import seed
from sklearn import metrics
from tensorboardX import SummaryWriter

torch.multiprocessing.set_sharing_strategy('file_system')


class Trainer:
    """
    Class `Trainer`: the class is used to train the model and optimize it
    by utilizing hyper-parameters tuning metrics (the metrics can be accuracy, auroc or
    weighted utility based)
    """

    def __init__(self, metric: str = '', model: AlexNet = None, learning_rate: float = 1e-5, scheduler: str = 'step',
                 task: str = 'abnormal', plane: str = 'sagittal', epochs: int = 20) -> None:
        self.model = model
        self.metric = metric
        self.task = task
        self.plane = plane
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.lr = learning_rate
        self.optimizer = optim.Adam(AlexNet().parameters(), lr=self.lr, weight_decay=1e-2)
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
        self.experiment_directory = os.path.join(
            self.experiment_path, self.experiment_name)
        if not os.path.exists(self.experiment_directory):
            os.makedirs(self.experiment_directory)
            os.makedirs(os.path.join(self.experiment_directory, 'models'))
            os.makedirs(os.path.join(self.experiment_directory, 'logs'))
            os.makedirs(os.path.join(self.experiment_directory, 'results'))
        self.log_root_path = (self.experiment_directory + f'/logs/{task}/{plane}') + datetime.now(). \
            strftime("%Y%m%d""-%H%M%S") + "/"
        os.makedirs(self.log_root_path)
        self.flush_history = 0

    @staticmethod
    def get_learning_rate(self):
        """
        `get_learning_rate`: returns the best learning rate from the parameters groups
        """
        for param in self.optimizer.param_groups:
            return param['lr']

    def train(self, writer: SummaryWriter, epoch: int):
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
            loss = F.binary_cross_entropy_with_logits(
                prediction, label, weight=weights)
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
                accuracy = metrics.balanced_accuracy_score(
                    y_label, y_prediction)
                if self.metric == 'accuracy':
                    auc = metrics.roc_auc_score(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training AUC', auc,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar(
                        'Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420:
                        break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {np.round(accuracy, 4)}")
                    writer.add_scalar('Training AUC per epoch', auc, epoch)
                    writer.add_scalar(
                        'Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_accuracy = np.round(accuracy, 4)

                    return training_loss, training_accuracy
                if self.metric == 'auc':
                    auc = metrics.auc(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training AUC', auc,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar(
                        'Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420:
                        break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {np.round(accuracy, 4)}")
                    writer.add_scalar('Training AUC per epoch', auc, epoch)
                    writer.add_scalar(
                        'Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_auc = np.round(auc, 4)

                    return training_loss, training_auc

                if self.metric == 'wu':
                    wu = ut.weighted_utility(y_label, y_prediction)
                    writer.add_scalar('Training Loss', loss_score,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar('Training WU', wu,
                                      epoch * len(self.train_loader) + i)
                    writer.add_scalar(
                        'Training Accuracy', accuracy, epoch * len(self.train_loader) + i)

                    if i == 420:
                        break
                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.train_loader)}"
                              f"\t Average training loss {np.round(np.mean(losses), 4)}"
                              f"\t Training WU {wu}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Training accuracy {accuracy}")
                    writer.add_scalar('Training WU per epoch', wu, epoch)
                    writer.add_scalar(
                        'Training accuracy per epoch', accuracy, epoch)

                    training_loss = np.round(np.mean(losses), 4)
                    training_wu = np.round(wu, 4)

                    return training_loss, training_wu
            except:
                auc = 0.5
                accuracy = 0.5
        return

    def fit(self, writer: SummaryWriter, epoch: int):
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
            loss = F.binary_cross_entropy_with_logits(
                prediction, label, weight=weights)

            loss_score = loss.item()
            losses.append(loss_score)

            score = torch.sigmoid(prediction)

            y_label.append(int(label[0]))
            y_prediction.append(score[0].item())
            y_class_predictions.append((score[0] > 0.5).float().item())

            try:
                accuracy = metrics.balanced_accuracy_score(
                    y_label, y_prediction)
                if self.metric == 'accuracy':
                    auc = metrics.roc_auc_score(y_label, y_prediction)
                    writer.add_scalar(
                        'Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation AUC', auc,
                                      epoch * len(self.test_loader) + i)
                    writer.add_scalar(
                        'Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar(
                            'Validation AUC per epoch', auc, epoch)
                        writer.add_scalar(
                            'Validation accuracy per epoch', accuracy, epoch)

                        val_loss = np.round(np.mean(losses), 4)
                        val_accuracy_epoch = np.round(accuracy, 4)

                        val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_label,
                                                                                                             y_class_predictions)
                        val_accuracy, val_sensitivity, val_specificity = np.round(val_accuracy, 4), np.round(
                            val_sensitivity, 4), np.round(val_specificity)

                        return val_loss, val_accuracy_epoch, val_accuracy, val_sensitivity, val_specificity
                if self.metric == 'auc':
                    auc = metrics.auc(y_label, y_prediction)
                    writer.add_scalar(
                        'Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation AUC', auc,
                                      epoch * len(self.test_loader) + i)
                    writer.add_scalar(
                        'Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation AUC {np.round(auc, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar(
                            'Validation AUC per epoch', auc, epoch)
                        writer.add_scalar(
                            'Validation accuracy per epoch', accuracy, epoch)

                        val_loss = np.round(np.mean(losses), 4)
                        val_accuracy_epoch = np.round(accuracy, 4)

                        val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_label,
                                                                                                             y_class_predictions)
                        val_accuracy, val_sensitivity, val_specificity = np.round(val_accuracy, 4), np.round(
                            val_sensitivity, 4), np.round(val_specificity)

                        return val_loss, val_accuracy_epoch, val_accuracy, val_sensitivity, val_specificity

                if self.metric == 'wu':
                    wu = ut.weighted_utility(y_label, y_prediction)
                    writer.add_scalar(
                        'Validation Loss', loss_score, epoch * len(self.test_loader) + i)
                    writer.add_scalar('Validation WU', wu,
                                      epoch * len(self.test_loader) + i)
                    writer.add_scalar(
                        'Validation Accuracy', accuracy, epoch * len(self.test_loader) + i)

                    if i % self.log_every == 0 and i > 0:
                        print(f"Epoch {epoch + 1} of {self.epochs}"
                              f"\t Batch {i} of {len(self.test_loader)}"
                              f"\t Average validation loss {np.round(np.mean(losses), 4)}"
                              f"\t Validation WU {np.round(wu, 4)}"
                              f"\t Learning rate {Trainer.get_learning_rate()}"
                              f"\t Validation accuracy {np.round(accuracy, 4)}")

                        writer.add_scalar(
                            'Validation AUC per epoch', auc, epoch)
                        writer.add_scalar(
                            'Validation accuracy per epoch', accuracy, epoch)

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

    def evaluate(self) -> None:
        """ Evaluate the model training """

        # Set random seed
        random.seed(42)
        torch.manual_seed(42)
        np.random.seed(42)
        torch.cuda.manual_seed_all(42)

        # create the directories to stpre the experiment checkpoint, logs and results
        exp_dir = self.experiment_directory
        log_dir = self.log_root_path

        writer = SummaryWriter(log_dir)

        # initialize the training variables
        best_val_loss = float('inf')
        best_val_auc = float(0)
        best_val_accuracy = float(0)
        best_val_wu = float(0)
        best_val_sensitivity = float(0)
        best_val_specificity = float(0)
        iteration_change_loss = 0

        # training
        t_start_training = time.time()

        self.device

        for epoch in range(self.epochs):
            self.get_learning_rate()

            t_start = time.time()

            if self.metric == 'accuracy':
                # train
                train_loss, train_acc = self.train(writer, epoch)

                # evaluate
                val_loss, val_auc, val_accuracy, val_sensitivity, val_specificity = self.fit(
                    writer, epoch)

                self.scheduler.step()

                t_end = time.time()
                delta = t_end - t_start

                print(f'Training Loss: {train_loss} - Train Accuracy: {train_acc} - Validation Los: {val_loss} - Validation AUC: {val_auc} - Validation Accuracy: {val_accuracy}'
                      + f'Elapsed time {delta}s')

                iteration_change_loss += 1
                print('-'*100)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_accuracy = val_accuracy
                    best_val_sensitivity = val_sensitivity
                    best_val_specificity = val_specificity

                    file_name = f'model_{self.metric}_{self.task}_{self.plane}.pt.tar'
                    for f in os.listdir(exp_dir + '/models/'):
                        if (self.task in f) and (self.plane in f):
                            os.remove(exp_dir + f'/models/{f}')
                        torch.save(self.model.state_dict(),
                                   exp_dir + f'/models/{file_name}')
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        iteration_change_loss = 0

                    if iteration_change_loss == 50:
                        print(
                            f'Early stopping after {iteration_change_loss} iterations any further decrease of the loss')
                        break

                # Save the results
                with open(os.path.join(exp_dir, 'results', f'model_{self.metric}_{self.task}_{self.plane}-results.csv'), 'w') as res_file:
                    fw = csv.writer(res_file, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    fw.writerow(['Loss Value', 'AUC best score', 'Accuracy best score',
                                'Sensitivity best score', 'Specifity best score'])
                    fw.writerow([best_val_loss, best_val_auc, best_val_accuracy,
                                best_val_sensitivity, best_val_specificity])
                    res_file.close()
                t_end_training = time.time()
                print(f'Training took {t_end_training - t_start_training} s')

            if self.metric == 'auc':
                # train
                train_loss, train_auc = self.train(writer, epoch)

                # evaluate
                val_loss, val_auc, val_accuracy, val_sensitivity, val_specificity = self.fit(
                    writer, epoch)

                self.scheduler.step()

                t_end = time.time()
                delta = t_end - t_start

                print(f'Training Loss: {train_loss} - Train AUC: {train_auc} - Validation Los: {val_loss} - Validation AUC: {val_auc} - Validation Accuracy: {val_accuracy}'
                      + f'Elapsed time {delta}s')

                iteration_change_loss += 1
                print('-'*100)

                if val_auc > best_val_auc:
                    best_val_auc = val_auc
                    best_val_accuracy = val_accuracy
                    best_val_sensitivity = val_sensitivity
                    best_val_specificity = val_specificity

                    file_name = f'model_{self.metric}_{self.task}_{self.plane}.pt.tar'
                    for f in os.listdir(exp_dir + '/models/'):
                        if (self.task in f) and (self.plane in f):
                            os.remove(exp_dir + f'/models/{f}')
                        torch.save(self.model.state_dict(),
                                   exp_dir + f'/models/{file_name}')
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        iteration_change_loss = 0

                    if iteration_change_loss == 50:
                        print(
                            f'Early stopping after {iteration_change_loss} iterations any further decrease of the loss')
                        break

                # Save the results
                with open(os.path.join(exp_dir, 'results', f'model_{self.metric}_{self.task}_{self.plane}-results.csv'), 'w') as res_file:
                    fw = csv.writer(res_file, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    fw.writerow(['Loss Value', 'AUC best score', 'Accuracy best score',
                                'Sensitivity best score', 'Specifity best score'])
                    fw.writerow([best_val_loss, best_val_auc, best_val_accuracy,
                                best_val_sensitivity, best_val_specificity])
                    res_file.close()

                t_end_training = time.time()
                print(f'training took {t_end_training - t_start_training} s')

            if self.metric == 'wu':
                # train
                train_loss, train_wu = self.train(writer, epoch)

                # evaluate
                val_loss, val_wu, val_accuracy, val_sensitivity, val_specificity = self.fit(
                    writer, epoch)

                self.scheduler.step()

                t_end = time.time()
                delta = t_end - t_start

                print(f'Training Loss: {train_loss} - Train WU: {train_wu} - Validation Los: {val_loss} - Validation WU: {val_auc} - Validation Accuracy: {val_accuracy}'
                      + f'Elapsed time {delta}s')

                iteration_change_loss += 1
                print('-'*100)

                if val_wu > best_val_wu:
                    best_val_wu = val_wu
                    best_val_accuracy = val_accuracy
                    best_val_sensitivity = val_sensitivity
                    best_val_specificity = val_specificity

                    file_name = f'model_{self.metric}_{self.task}_{self.plane}.pt.tar'
                    for f in os.listdir(exp_dir + '/models/'):
                        if (self.task in f) and (self.plane in f):
                            os.remove(exp_dir + f'/models/{f}')
                        torch.save(self.model.state_dict(),
                                   exp_dir + f'/models/{file_name}')
                    if val_loss < best_val_loss:
                        best_val_loss = val_loss
                        iteration_change_loss = 0

                    if iteration_change_loss == 50:
                        print(
                            f'Early stopping after {iteration_change_loss} iterations any further decrease of the loss')
                        break

                # Save the results
                with open(os.path.join(exp_dir, 'results', f'model_{self.metric}_{self.task}_{self.plane}-results.csv'), 'w') as res_file:
                    fw = csv.writer(res_file, delimiter=',',
                                    quotechar='|', quoting=csv.QUOTE_MINIMAL)
                    fw.writerow(['Loss Value', 'WU best score', 'Accuracy best score',
                                'Sensitivity best score', 'Specifity best score'])
                    fw.writerow([best_val_loss, best_val_wu, best_val_accuracy,
                                best_val_sensitivity, best_val_specificity])
                    res_file.close()

                t_end_training = time.time()
                print(f'Training took {t_end_training - t_start_training} s')


if __name__ == "__main__":
    writer = SummaryWriter()
    Trainer('auc').fit(writer, 20)
    
