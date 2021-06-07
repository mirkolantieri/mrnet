import numpy as np
import pandas as pd
from scipy.stats import entropy  # entropy lib


class Cases:

    def __init__(self, csv) -> None:
        self.csv = pd.read_csv(csv)
    
    def find_low_agreement(self) -> pd:
        """  """
        # read agreement csv file
        agree = self.csv
        # drop the last column
        agree = agree.drop(columns=["Unnamed: 0", "AI"]) 

        # find the entropy to be applied for the agreement, then sort the dataframe (get 25 cols)
        id = entropy(agree.apply(lambda index: index.value_counts(normalize=True), 
            axis=1).fillna(0), axis=1).argsort()[::-1][:25] 
        id = np.sort(id)

        # print the id labels and 
        # output the table with the corresponding values
        print(f'{id}\n') 
        print(f'{agree.iloc[id].sort_index()}\n')

        return agree.iloc[id].sort_index()

    def find_most_complex(self) -> pd:
        """  """

        # read complex csv file
        cmplx = self.csv

        # assign the index column
        # sort the dataframe then choose the first 25 complex cases

        id = cmplx.mean(axis=1).values.argsort()[::-1][:25]

        # print the id labels and 
        # output the table with the corresponding values
        print(f'{id}\n') 
        print(f'{cmplx.iloc[id].sort_index()}\n')

        return cmplx.iloc[id].sort_index()

if __name__ == '__main__':
    Cases('./cases/classes.csv').find_low_agreement()
    Cases('./cases/difficolta.csv').find_most_complex()

