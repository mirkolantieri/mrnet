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
from prediction import extract_predictions
import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from torch._C import device
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

    def __init__(self, model_path: str, task: str, plane: str, root_dir: str, store:str ) -> None:
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.task = task
        self.tasks = ['acl', 'meniscus', 'abnormal']
        self.plane = plane
        self.planes = ['axial', 'coronal', 'sagittal']
        self.model_name = list(filter(lambda name: self.task in name and
        self.plane in name, os.listdir(model_path) ))[0]
        self.model_path = f"{model_path}/{self.model_name}"
        self.model = AlexNet()
        self.logreg = LogisticRegression(solver='lbfgs')
        self.root_dir = root_dir
        self.store = store

    
    def __extract_prediction(self, train: bool = False):
        """ `extract_prediction`: helper method that extract the predictions
        from the tensor feed.

        args: self, 
        train:bool=False : boolean variable that enable the extraction
        on the validation set if false or the training if true

        returns: predictions and the target label
        """
        assert self.task in self.tasks
        assert self.plane in self.planes

        # Array lists that will store the predictions and the train_labels
        predictions, train_labels = [],[]
            
        # Load the model and enable the computation device (cpu or gpu in case of available Nvidia graphics)
        self.model.load_state_dict(torch.load(self.model_path))
        self.model.to(self.device)
        # Initiate the evaluation
        self.model.eval()

        # Initiate the dataset and send it to the loader
        set = MRDataset(self.root_dir,self.task, self.plane, train=train)
        loader = torch.utils.data.DataLoader(set, batch_size=1, shuffle=False, num_workers=2, drop_last=False)

        while torch.no_grad():
            i=0
            for img, label, _ in tqdm.tqdm(loader):
                if i >= 420: break
                img = img.to(device)
                logit = self.model(img)
                prediction = torch.softmax(logit)
                predictions.append(prediction[0].item())
                train_labels.append(label[0].item())
            
        # Return the predictions
        return np.array(predictions), np.array(train_labels)


    def predict(self, complex: bool = False):

        final_results = {}

        for self.task in self.tasks:
            train_results = {}
            test_results = {}

            # Train the logistic regressor model (transfer learning)
            for plane in self.planes:
                train_preds, train_labels = self.__extract_prediction(train=True)
                test_preds, test_labels = self.__extract_prediction(train=complex)

                train_results['train_labels'] = train_labels
                train_results[plane] = train_preds
                test_results['test_labels'] = test_labels
                test_results[plane] = test_preds

            # Slice the prediction to a numpy array with 3 channels (due to the number of planes we have)
            X_train = np.zeros((len(train_preds), 3))
            X_train[:,0] = train_results['axial'] 
            X_train[:,1] = train_results['coronal']
            X_train[:,2] = train_results['sagittal']

            X_test = np.zeros((len(test_preds), 3))
            X_test[:,0] = test_results['axial'] 
            X_test[:,1] = test_results['coronal']
            X_test[:,2] = test_results['sagittal']

            # Create the array with the train and test labels
            y_train = train_labels
            y_test = test_labels

            # Fit the model
            self.logreg.fit(X_train, y_train)
            probas = self.logreg.predict_proba(X_test)[:,1]
            probas = (probas>0.5).astyoe(np.float32)

            # Calculate the AUC-score and Weighted Utility
            auc = metrics.roc_auc_score(y_train, probas)
            wu = ut.weighted_utility(y_train, probas)

            print(f"Current task: {self.task}\nAUC Score {auc} - Weighted Utility {wu}")

            accuracy, sensitivity, specificity = ut.accuracy_sensitivity_specificity(y_test, probas)
            final_results[self.task] = [auc, accuracy, sensitivity, specificity, wu]

            y_train = pd.DataFrame(y_train)
            probas = pd.DataFrame(probas)

            if complex:
                y_train.to_csv(f"./{self.store}/complex-{self.task}-label.csv", sep=',')
                probas.to_csv(f"./{self.store}/complex-{self.task}-predictions.csv", sep=',')
            
            y_train.to_csv(f"./{self.store}/{self.task}-label.csv", sep=',')
            probas.to_csv(f"./{self.store}/{self.task}-predictions.csv", sep=',')
        
        experiment = self.model_path.split('/')[:-2]
        if complex:
            with open(os.path.join(*experiment, 'results', f'complex-auc-results.csv'), 'w') as res_file:
                fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
                fw.writerow(['Tear', 'AUC', 'Accuracy', 'Sensitivity', 'Specifity', 'Weighted Utility'])
                for ck in final_results.keys():
                    fw.writerow([f'{ck}'] + [str(val) for val in final_results[ck]])

        with open(os.path.join(experiment, 'results', f'auc-results.csv'), 'w') as res_file:
            fw = csv.writer(res_file, delimiter=',', quotechar='|', quoting=csv.QUOTE_MINIMAL)
            fw.writerow(['Tear', 'AUC', 'Accuracy', 'Sensitivity', 'Specifity', 'Weighted Utility'])
            for ck in final_results.keys():
                fw.writerow([f'{ck}'] + [str(val) for val in final_results[ck]])

        return
