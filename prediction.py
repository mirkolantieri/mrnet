# 
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
# 
# prediction.py : script responsable for the prediction of the 0/1 (positive-negative) rates from the models
# The file contains the respective methods: extract_predictions
# 

# Importing libraries

import argparse
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

import utils as ut
from loader import MRDataset
from model import AlexNet

torch.multiprocessing.set_sharing_strategy('file_system')

parser = argparse.ArgumentParser()
parser.add_argument('--path', type=str, required=True)
parser.add_argument('--store', type=str, help='the path where to store the data', required=True)
parser.add_argument('--data', type=str)
parser.add_argument('--seed', type=int, default=42)

args = parser.parse_args()

random.seed(args.seed)
torch.manual_seed(args.seed)
np.random.seed(args.seed)
torch.cuda.manual_seed_all(args.seed)

def extract_predictions(task, plane, path_to_models, train=True):
    """ 
    `extract_predictions`: the method extracts the prediction from the pretrained models

    args:
        task: the tear to be analized (acl, meniscus, abnormal)
        plane: the plane where the tear occured (for example: axial, coronal, sagittal)
        path_to_models: the path where are stored the trained models
        trains=True: boolean to indicate if the loader needs to be from the training set or the validation

    """
    assert task in ['acl', 'meniscus', 'abnormal']
    assert plane in ['axial', 'coronal', 'sagittal']
    
    # Initialize the models and filter by the tear
    models = os.listdir(path_to_models)
    model_name = list(filter(lambda name: task in name and plane in name, models))[0]
    model_path = f'{path_to_models}/{model_name}'

    # Select the gpu or the cpu for the tensor compilation
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else: 
        device = torch.device('cpu')

    model = AlexNet()

    # Load the model
    mrnet = torch.load(model_path)
    model.load_state_dict(mrnet)
    model.to(device)
    model.eval()
    
    # Create the traning set and send to the loader
    dataset = MRDataset(args.data, task, plane, train=train)
    
    loader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

    # Create the array list to store the predictions and labels from the compiled model
    predictions = []
    labels = []

    skip = { 280,97,77,394,259,115,314,123,100,279,150,30,309,381,231,94,146,230,325,339,258,137,193,298,207 }

    # While compiling without gradient, add each single item from the labels and prediction
    # to the above defined array lists 
    # and then return it 
    i = 0
    with torch.no_grad():
        for image, label, _ in tqdm.tqdm(loader):
            i +=1 
            image = image.to(device)
            logit = model(image)
            prediction = torch.sigmoid(logit)
            predictions.append(prediction[0].item())
            labels.append(label[0].item())
            if i == 420: break
            if i in skip: pass

    return predictions, labels


final_results_val = {}

for task in ['acl', 'meniscus', 'abnormal']:
    results = {}

    # Train a logistic regressor model
    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, args.path)
        results['labels'] = labels
        results[plane] = predictions
    
    # Transform the predictions to a numpy array with 3 channels 
    # (since we have 3 planes {axial, coronal, saggital})
    X = np.zeros((len(predictions), 3))
    X[:, 0] = results['axial']
    X[:, 1] = results['coronal']
    X[:, 2] = results['sagittal']

    # Create an numpy array having the labels
    y = np.array(labels)

    # Fit the model
    logreg = LogisticRegression(solver='lbfgs')
    logreg.fit(X, y)

    # Create a logistic regressor for the validation test
    results_val = {}

    for plane in ['axial', 'coronal', 'sagittal']:
        predictions, labels = extract_predictions(task, plane, args.path, train=False)
        results_val['labels'] = labels
        results_val[plane] = predictions

    X_val = np.zeros((len(predictions), 3))
    X_val[:, 0] = results_val['axial']
    X_val[:, 1] = results_val['coronal']
    X_val[:, 2] = results_val['sagittal']

    y_val = np.array(labels)

    # Predict the validation using the trained model, return
    # the probabilities which are grater then 0.5 (rounding them to 1)
    # then save to a csv file

    y_pred = logreg.predict_proba(X_val)[:, 1]
    y_class_preds = pd.DataFrame((y_pred > 0.5).astype(np.float32))
    auc = metrics.roc_auc_score(y_val, y_pred)
    print(f'{task} AUC: {auc}')
    y_class_preds.to_csv(f'./{args.store}/{task}-prediction.csv', sep=',') # save the predicts of  the final result considering each plane 

    

    plt.figure(0).clf()

    fpr, tpr, thresh = metrics.roc_curve(y_val, y_pred)
    plt.plot(fpr,tpr,label=f"Task {task}, auc="+str(auc))
    plt.title(f'ROC/AUROC graph')
    plt.xlabel('fpr')
    plt.ylabel('tpr')
    plt.legend(loc=0)
    plt.savefig(f'./{args.store}/roc-{task}.jpg')

    accuracy, sensitivity, specificity = ut.accuracy_sensitivity_specificity(y_val, y_class_preds)
    final_results_val[task] = [auc, accuracy, sensitivity, specificity]

exp_dir = args.path.split('/')[:-2]

# Save the obtained AUC results to a csv file 
with open(os.path.join(*exp_dir, 'results', f'auc-results.csv'), 'w') as res_file:
    fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
    fw.writerow(['Tear', 'AUC', 'Accuracy', 'Sensitivity', 'Specifity'])
    for ck in final_results_val.keys():
        fw.writerow([f'{ck}'] + [str(val) for val in final_results_val[ck]])

