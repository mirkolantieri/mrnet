#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# utils.py : script containing helpful methods
# The file contains the respective methods:
# random_rotate
# random_shift
# random_flip
# accuracy_sensitivity_specificity
# preprocess_image
# deprocess_image
# show_cam_on_image

# Importing libraries



import random
import cv2
import numpy as np
from numpy.core.fromnumeric import mean
from numpy.lib.function_base import average
from numpy.random import exponential
import pandas as pd
from scipy.ndimage.measurements import median
import sklearn.metrics as metrics
import torch
from scipy.ndimage import shift
from skimage.transform import rotate
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import confusion_matrix
from torchvision.transforms import Compose, Normalize, ToTensor


def random_rotate(array, max_angle):
    """
    `random_rotate`: rotates the image frame from different angles

    args:
        array: the image provided as float array
        max_angle: the maximum angle to which rotate the image frame 
    """
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)

        random_angle = random.randint(-max_angle, max_angle)

        for i in range(array.shape[0]):
            array_out[i] = rotate(array[i], random_angle, preserve_range=True)

        return array_out
    else:
        return array


def random_shift(array, max_shift):
    """
    `random_shift`: shifts randomply the pixels of an image frame

    args: 
        array: the image provided as float array
        max_shift: the maximum range for shifting the pixels 
    """
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)

        random_x = random.randint(-max_shift, max_shift)
        random_y = random.randint(-max_shift, max_shift)

        for i in range(array.shape[0]):
            array_out[i] = shift(array[i], (random_x, random_y))

        return array_out
    else:
        return array


def random_flip(array):
    """ 
    `random_flip`: randomly flips the image frame

    args:
        array: the image provided as float array
    """
    if random.random() > 0.5:
        array_out = np.zeros(array.shape)

        for i in range(array.shape[0]):
            array_out[i] = np.fliplr(array[i])

        return array_out
    else:
        return array


def accuracy_sensitivity_specificity(y_trues, y_preds):
    """ 
    `accuracy_sensitivity_specificity`: calculate the AUROC

    args:
        y_trues: the array containing the true labels
        y_preds: the array containing the predicted labels
    """
    cm = confusion_matrix(y_trues, y_preds)
    tn, fp, fn, tp = cm.ravel()
    total = sum(sum(cm))
    # accuracy = (cm[0,0] + cm[1,1]) / total
    # sensitivity = cm[1,1] / (cm[1,1] + cm[1,0])
    # specificity = cm[0,0] / (cm[0,0] + cm[0,1])
    accuracy = (tp + tn) / total
    sensitivity = tp / (tp + fn)
    specificity = tn / (tn + fp)

    return accuracy, sensitivity, specificity


def weighted_utility(y_true, y_preds, gamma=1):
    """
    `weighted_utility`: calculate the weighted utility

    args:
        y_trues: the array containing the true labels
        y_preds: the array containing the predicted labels
        gamma: coefficient (assuming the default value is 1
            sigma(y_preds[i] | thresholds[i]) = 1 if y_preds[1] > thresholds[i] else 0
        )
    """ 
    # Ignore the division with small numbers
    np.seterr(divide='ignore', invalid='ignore')
    
    # Set array to 1D
    y_true = np.array(y_true).reshape(-1, 1)
    y_true = np.array(y_true).reshape(-1, 1)
    
    # Compute precision recall and thresholds with sklearn metrics
    precision, recall, thresholds = metrics.precision_recall_curve(y_true, y_preds)

    # Train a simple linear model to obtain the most important features
    r = permutation_importance(
        Ridge(alpha=1e-2).fit(y_true, y_preds),
        y_true,
        y_preds,
        n_repeats=10,
        random_state=999,
    )

    # Calculate sigma(h|t)
    # and convert the thresholds to mean for auto-bias
    s = 0
    tau = mean(thresholds)
    pos = float(max(r.importances_mean))

    for preds in y_preds:
        for r_i in r.importances:
            if preds <= tau and preds >= gamma * tau:
                if gamma == 1: pass
                s = (preds - (gamma * tau)) / float((1 - gamma) * tau)
            if preds > tau: s = 1
            else: s = 0
            A = sum(r_i * s)
            B = sum(r_i * (tau / 1 - tau) * s)
    
    # Calculate the weighted utility
    wU = pow(pos, -1) * A - pow(pos, -1) * B
    

    return wU

def preprocess_image(img: np.ndarray, mean=None, std=None) -> torch.Tensor:
    """ 
    `preprocess_image`: preprocess an image frame by converting it to a torch tensor

    args:
        img: the image provided as nd.array
        mean=None: the list of mean values to be used for processing
        std: standard deviation values to be used for the processing
    """
    if std is None:
        std = [0.5, 0.5, 0.5]
    if mean is None:
        mean = [0.5, 0.5, 0.5]

    preprocessing = Compose([ToTensor(), Normalize(mean=mean, std=std)])

    return preprocessing(img.copy()).unsqueeze(0)


def deprocess_image(img):
    """
    `deprocess_image` : converts the float array to an image frame  

    args:
        img: the float array
    """
    img = img - np.mean(img)
    img = img / (np.std(img) + 1e-5)
    img = img * 0.1
    img = img + 0.5
    img = np.clip(img, 0, 1)
    return np.uint8(img * 255)


def show_cam_on_image(img: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    `show_cam_on_image` : returns the class activation maps of an image
    
    args: 
        img: the image provided as an nd.array
        mask: the mask filter as an nd.array
    """
    heatmap = cv2.applyColorMap(np.uint8(255 * mask), cv2.COLORMAP_JET)
    heatmap = np.float32(heatmap) / 255
    cam = heatmap + np.float32(img)
    cam = cam / np.max(cam)
    return np.uint8(255 * cam)


weighted_utility(np.array([1, 1, 1, 1, 0, 1]), np.array([1, 0, 1, 1, 0, 0]), 1)