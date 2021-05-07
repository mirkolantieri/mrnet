# MRI Deep Learning Model
## Project for master thesis

*author* : **Mirko Lantieri** <br>
*code name*: **utility first** <br>

---
# Table of contents
1. [Introduction](#introduction)
2. [Dataset](#dataset)
    1. [25th most difficult cases](#difficult_cases)
    2. [25th most highest disagreement cases](#disagreement)
3. [Dependencies](#dependencies)


---
# 1. Introduction <a name="introduction"></a>
In this project will be created a potential AI based on deep learning techniques: 
the idea behind is to create a neural network and train it with cases from
the MRNet dataset from Stanford in order to predict the cases of knee MRI images
from patients having problems or hidden disease. The main interest is to 
comprehend that if for a given MRI scan image, the prediction from the
model is helpful for a radiologist in terms of utility instead of accuracy. We are interested in creating 2 AI models consisting of 
Convolutional Neural Networks:
* one generic model based on a simplistic implementation of a CNN and AlexNet architecture
* one model having implemented in-structure hyperparameter tuning implementation, guided by an utility-based method

 
After the implementation of the two main models we will then implement the
activation map for each medical case.

---

# 2. Dataset <a name="dataset"></a>
The dataset collected is from the [MRNet Dataset](https://stanfordmlgroup.github.io/competitions/mrnet/) provided by [Stanford ML Group](https://stanfordmlgroup.github.io/)
The MRNet dataset consists of 1,370 knee MRI exams performed at Stanford
 University Medical Center. The dataset contains 1,104 (80.6%) abnormal 
 exams, with 319 (23.3%) ACL tears and 508 (37.1%) meniscal tears.

 The structure of the dataset is as follows:



    data/
        train/
            axial/
            coronal/
            sagittal/
        valid/
            axial/
            coronal/
            sagittal/
        train-acl.csv
        train-abnormal.csv
        train-meniscus.csv
        valid-acl.csv
        valid-abnormal.csv
        valid-meniscus.csv
    
You can download the dataset from the above link

<br>

## 2.1 The 25th Most Difficult Cases <a name="difficult_cases"></a>
This is 

<br>

## 2.2 The 25th Most Highest Disagreement Cases <a name="disagreement"></a>
The second paragraph text

---

# 3. Installing dependencies <a name="dependencies"></a>
Before running the script file, you should install first the *(probably missing)*
dependencies which can be found under 
`requirements.txt`. Here is how to install:

**Terminal (Mac/Linux)**
Type: `pip3 install -r requirements.txt` 


**Powershell/Command Prompt (Windows)**
Type: `pip install -r requirements.txt`


