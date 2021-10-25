#
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
#
# loader.py : script responsible for loading the dataset path
# The file contains the respective class: MRDataset(data.Dataset);
# inherits an object from the Dataset parent class implemented by PyTorch.org
# Importing libraries

import numpy as np
import pandas as pd
import torch
import torch.utils.data as data


import helper.utils as ut


class MRDataset(data.Dataset):
    def __init__(self, root_dir, tear, plane, train=True, weights=None):
        super(MRDataset, self).__init__()
        self.tear = tear
        self.plane = plane
        self.root_dir = root_dir
        self.train = train
        transform = None
        if self.train:
            self.folder_path = self.root_dir + "train/{0}/".format(plane)
            self.records = pd.read_csv(
                self.root_dir + "train-{0}.csv".format(tear),
                header=None,
                names=["id", "label"],
            )
        else:
            self.folder_path = self.root_dir + "valid/{0}/".format(plane)
            self.records = pd.read_csv(
                self.root_dir + "valid-{0}.csv".format(tear),
                header=None,
                names=["id", "label"],
            )

        self.records["id"] = self.records["id"].map(
            lambda i: "0" * (4 - len(str(i))) + str(i)
        )
        self.paths = [
            self.folder_path + filename + ".npy"
            for filename in self.records["id"].tolist()
        ]
        self.labels = self.records["label"].tolist()

        self.transform = transform
        if weights is None:
            neg_weight = np.mean(self.labels)
            self.weights = torch.FloatTensor([neg_weight, 1 - neg_weight])
        else:
            self.weights = torch.FloatTensor(weights)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        array = np.load(self.paths[index])

        label = torch.FloatTensor([self.labels[index]])

        weight = torch.FloatTensor([self.weights[self.labels[index]]])

        if self.train:
            # data augmentation
            array = ut.random_shift(array, 25)
            array = ut.random_rotate(array, 25)
            array = ut.random_flip(array)

        # data standardization
        array = (array - 58.09) / 49.73
        array = np.stack((array,) * 3, axis=1)

        array = torch.FloatTensor(array)  # array size is now [S, 224, 224, 3]

        return array, label, weight
