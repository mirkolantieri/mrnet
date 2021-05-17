# 
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
# 
# train_auc.py : script responsable for the training and evaluation of the AlexNet CNN model based on auc.
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

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn import metrics
from tensorboardX import SummaryWriter
from tqdm import tqdm

import utils as ut
from loader import MRDataset
from model import AlexNet

torch.multiprocessing.set_sharing_strategy('file_system')

def train_model(model, train_loader, epoch, num_epochs, optimizer, writer, current_lr, device, log_every=100):
    """  
    `train_model`: train the model for a given number of epochs, learning rate, and an optimizer for the network model

    args:
        model: the model used for the training
        train_loader: the loader path of the training set
        epoch: the current epoch under performing of the train
        num_epochs: the total number of epochs to perform the training
        optimizer: the hyperparameter tuning configuration for the optimization of the model
        writer: the writer for the tensorboar
        current_lr: the current learning rate during the epoch
        device: the device to be used (gpu or cpu) for the model training
        log_every: the ammount of logs to be processed
    """
    model.train()
    model = model.to(device)

    y_preds = []
    y_trues = []
    y_class_preds = []
    losses = []

    for i, (image, label, weight) in enumerate(train_loader):

        image = image.to(device)
        label = label.to(device)
        weight = weight.to(device)

        prediction = model(image.float())

        loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weight)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        y_preds.append(probas[0].item())
        y_class_preds.append((probas[0] > 0.5).float().item())


        try:
            accuracy = metrics.precision_score(y_trues, y_preds)
            auc = metrics.roc_auc_score(y_trues, y_class_preds)
        except:
            auc = 0.5
            accuracy = 0.5

        writer.add_scalar('Train/Loss', loss_value, epoch * len(train_loader) + i)
        writer.add_scalar('Train/AUC', auc, epoch * len(train_loader) + i)
        writer.add_scalar('Train/Accuracy', accuracy, epoch * len(train_loader) + i)

        if i == 420:
            break

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ]| Average Train Loss {4} | Train AUC : {5} | LR : {6} | Accuracy {7}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(train_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr,
                      np.round(accuracy, 4)
                  )
                  )

    writer.add_scalar('Train/AUC per epoch', auc, epoch)
    writer.add_scalar('Train/Accuracy per epoch', accuracy, epoch)

    train_loss_epoch = np.round(np.mean(losses), 4)
    train_auc_epoch = np.round(auc, 4)

    return train_loss_epoch, train_auc_epoch


def evaluate_model(model, val_loader, epoch, num_epochs, writer, current_lr, device, log_every=20):
    """
    `evaluate_model`: evaluate the model for a given number of epochs, learning rate, and an optimizer for the network model

    args:
        model: the model used for the training
        val_loader: the loader path of the validation set
        epoch: the current epoch under performing of the train
        num_epochs: the total number of epochs to perform the training
        writer: the writer for the tensorboar
        current_lr: the current learning rate during the epoch
        device: the device to be used (gpu or cpu) for the model training
        log_every: the ammount of logs to be processed 
    """

    model.eval()

     # Select the gpu or the cpu for the tensor compilation
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


    y_trues = []
    y_preds = []
    y_class_preds = []
    losses = []

    for i, (image, label, weight) in enumerate(val_loader):
        image = image.to(device)
        label = label.to(device)
        weight = weight.to(device)

        prediction = model.forward(image.float())

        loss = F.binary_cross_entropy_with_logits(prediction, label, weight=weight)

        loss_value = loss.item()
        losses.append(loss_value)

        probas = torch.sigmoid(prediction)

        y_trues.append(int(label[0]))
        
        y_preds.append(probas[0].item())
        y_class_preds.append((probas[0] > 0.5).float().item())


        try:
            auc = metrics.roc_auc_score(y_trues, y_preds)
            accuracy = metrics.precision_score(y_trues, y_class_preds)
        except:
            auc = 0.5
            accuracy = 0.5
            
        writer.add_scalar('Val/Loss', loss_value, epoch * len(val_loader) + i)
        writer.add_scalar('Val/AUC', auc, epoch * len(val_loader) + i)
        writer.add_scalar('Val/Accuracy', accuracy, epoch * len(val_loader) + i)

        if (i % log_every == 0) & (i > 0):
            print('''[Epoch: {0} / {1} |Single batch number : {2} / {3} ] | Average Validation Loss {4} | Validation AUC : {5} | LR : {6} | Validation Accuracy {7}'''.
                  format(
                      epoch + 1,
                      num_epochs,
                      i,
                      len(val_loader),
                      np.round(np.mean(losses), 4),
                      np.round(auc, 4),
                      current_lr,
                      np.round(accuracy, 4)
                  )
                  )

    writer.add_scalar('Val/AUC per epoch', auc, epoch)
    writer.add_scalar('Val/Accuracy per epoch', accuracy, epoch)

    val_loss_epoch = np.round(np.mean(losses), 4)
    val_auc_epoch = np.round(auc, 4)

    val_accuracy, val_sensitivity, val_specificity = ut.accuracy_sensitivity_specificity(y_trues, y_class_preds)
    val_accuracy = np.round(val_accuracy, 4)
    val_sensitivity = np.round(val_sensitivity, 4)
    val_specificity = np.round(val_specificity, 4)

    return val_loss_epoch, val_auc_epoch, val_accuracy, val_sensitivity, val_specificity


def get_lr(optimizer):
    """
     `get_lr`: returns the best learning rate from the optimized parameters groups
    """
    for param_group in optimizer.param_groups:
        return param_group['lr']

def run(args):
    """ 
     `run`: runs the script by parsing the arguments via the terminal
    """
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)

    # create dirs to store experiment checkpoints, logs, and results
    exp_dir_name = args.experiment
    exp_dir = os.path.join('experiments', exp_dir_name)
    if not os.path.exists(exp_dir):
        os.makedirs(exp_dir)
        os.makedirs(os.path.join(exp_dir, 'models'))
        os.makedirs(os.path.join(exp_dir, 'logs'))
        os.makedirs(os.path.join(exp_dir, 'results'))

    log_root_folder = exp_dir + "/logs/{0}/{1}/".format(args.task, args.plane)
    if args.flush_history == 1:
        objects = os.listdir(log_root_folder)
        for f in objects:
            if os.path.isdir(log_root_folder + f):
                shutil.rmtree(log_root_folder + f)

    now = datetime.now()
    logdir = log_root_folder + now.strftime("%Y%m%d-%H%M%S") + "/"
    os.makedirs(logdir)

    writer = SummaryWriter(logdir)

    # create training and validation set
    train_dataset = MRDataset('./data/', args.task, args.plane, train=True)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1, shuffle=True, num_workers=2, drop_last=False)

    validation_dataset = MRDataset('./data/', args.task, args.plane, train=False)
    validation_loader = torch.utils.data.DataLoader(validation_dataset, batch_size=1, shuffle=-True, num_workers=2, drop_last=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # create the model
    mrnet = AlexNet()
    
    mrnet = mrnet.to(device)

    optimizer = optim.Adam(mrnet.parameters(), lr=args.lr, weight_decay=0.01)

    if args.lr_scheduler == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=.3, threshold=1e-4, verbose=True)
    elif args.lr_scheduler == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=3, gamma=args.gamma)

    best_val_loss = float('inf')
    best_val_auc = float(0)
    best_val_accuracy = float(0)
    best_val_sensitivity = float(0)
    best_val_specificity = float(0)

    num_epochs = args.epochs
    iteration_change_loss = 0
    patience = args.patience
    log_every = args.log_every

    t_start_training = time.time()

    # train and test loop
    for epoch in range(num_epochs):
        current_lr = get_lr(optimizer)

        t_start = time.time()
        
        # train
        train_loss, train_auc = train_model(mrnet, train_loader, epoch, num_epochs, optimizer, writer, current_lr, device, log_every)
        
        # evaluate
        val_loss, val_auc, val_accuracy, val_sensitivity, val_specificity = evaluate_model(mrnet, validation_loader, epoch, num_epochs, writer, current_lr, device)

        if args.lr_scheduler == 'plateau':
            scheduler.step(val_loss)
        elif args.lr_scheduler == 'step':
            scheduler.step()

        t_end = time.time()
        delta = t_end - t_start

        print("Train Loss : {0} | Train AUC {1} | Validation Loss {2} | Validation AUC {3} | elapsed time {4} s".format(
            train_loss, train_auc, val_loss, val_auc, delta))

        iteration_change_loss += 1
        print('-' * 30)

        if val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_accuracy = val_accuracy
            best_val_sensitivity = val_sensitivity
            best_val_specificity = val_specificity
            if bool(args.save_model):
                file_name = f'model_{args.prefix_name}_{args.task}_{args.plane}.pt'
                for f in os.listdir(exp_dir + '/models/'):
                    if (args.task in f) and (args.plane in f) and (args.prefix_name in f):
                        os.remove(exp_dir + f'/models/{f}')
                torch.save(mrnet.state_dict(), exp_dir + f'/models/{file_name}')

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            iteration_change_loss = 0

        if iteration_change_loss == patience:
            print('Early stopping after {0} iterations without the decrease of the val loss'.
                  format(iteration_change_loss))
            break
        
    # save results to csv file
    with open(os.path.join(exp_dir, 'results', f'model_{args.prefix_name}_{args.task}_{args.plane}-results.csv'), 'w') as res_file:
        fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        fw.writerow(['LOSS', 'AUC-best', 'Accuracy-best', 'Sensitivity-best', 'Specifity-best'])
        fw.writerow([best_val_loss, best_val_auc, best_val_accuracy, best_val_sensitivity, best_val_specificity])
        res_file.close()

    t_end_training = time.time()
    print(f'training took {t_end_training - t_start_training} s')


def parse_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('-t', '--task', type=str, required=True,
                        choices=['abnormal', 'acl', 'meniscus'])
    parser.add_argument('-p', '--plane', type=str, required=True,
                        choices=['sagittal', 'coronal', 'axial'])
    parser.add_argument('--data-path', type=str)
    parser.add_argument('--prefix_name', type=str, required=True)
    parser.add_argument('--experiment', type=str, required=True)
    parser.add_argument('--augment', type=int, choices=[0, 1], default=1)
    parser.add_argument('--lr_scheduler', type=str,
                        default='plateau', choices=['plateau', 'step'])
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--gamma', type=float, default=0.5)
    parser.add_argument('--epochs', type=int, default=20)
    parser.add_argument('--lr', type=float, default=1e-5)
    parser.add_argument('--flush_history', type=int, choices=[0, 1], default=0)
    parser.add_argument('--save_model', type=int, choices=[0, 1], default=1)
    parser.add_argument('--patience', type=int, default=50)
    parser.add_argument('--log_every', type=int, default=100)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse_arguments()
    run(args)
