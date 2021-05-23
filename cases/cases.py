# 
# 2021 (c) by **Mirko Lantieri**
# All rights reserved.
# 
# cases.py : script responsable for finding the 25th most difficult cases and 25th cases with the most highest disagrement


# Import libraries

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import time

from scipy.stats import entropy

# Intilize the time and read the csv dataframe

init_time = time.thread_time()
classes_df = pd.read_csv('./cases/classes.csv')
diff_df = pd.read_csv('./cases/difficolta.csv')

# Drop the last column which is redundant
classes_df = classes_df.drop(columns=["Unnamed: 0", "AI"])

# Sort the id by comparing it with the entropy: the higher the entropy the lowest the disagreement
id = entropy(classes_df.apply(lambda val: val.value_counts(normalize=True),
                              axis=1).fillna(0), axis=1).argsort()[::-1][:25]

print('Id of the cases with the most lowest agreement [highest disagreement]: ', id)

low_agreement = classes_df.iloc[id]
low_agreement.to_csv('./cases/casi_agreement.csv',
                     header=True, index=True, sep=',', mode='a')

# Sort the id of the most difficult cases by using simple mean
id = diff_df.mean(axis=1).values.argsort()[::-1][:25]

print("Id of the most difficult cases: ", id)

most_diff = diff_df.iloc[id]
most_diff.to_csv('./cases/casi_difficili.csv', header=True,
                 index=True, sep=',', mode='a')

end_time = time.thread_time()

print(f'Compilation times: {float(end_time-init_time)}s')
