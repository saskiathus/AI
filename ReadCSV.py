#https://scikit-learn.org/stable/datasets/index.html

#Use pandas to avoid the commas issue
import pandas as pd
import numpy as np
import random as rd
import os
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

encoder_name = "enc"

def convert_CsvToData(name, TrainingData = True, index_start = 0, index_end = -1, Sort = False):
    """Extract all data from a csv file. Each column is delemited by a comma.

    Parameters
    ----------
    name: Name or relative path of the file with the extension.

    TrainingData: True if the file contain Target values.
        Otherwise it consider to contain only Inputs to make prediction.

    Returns
    -------
    inData, outData: tuple containing a DataFrame of Inputs of the datafile and
        an array of Target. The array of Target is full of 0 if TrainingData is set False.
    """
    #Extract data from csv file, and put Nan value to 0
    rawData = pd.read_csv(name, delimiter = ',', dtype={"dclkVerticals": object})
    #del rawData['ip']
    rawData.drop('ip', axis=1, inplace=True)
    rawData = rawData.fillna(0)

    #Neural Networks doesn't support string, so convert them => Encoder
    if TrainingData:
        if not(os.path.isfile(encoder_name)):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(rawData.iloc[:,:-1])
            pickle.dump(enc, open(encoder_name, "wb"))
        else:
            enc = pickle.load(open(encoder_name,"rb"))
        if Sort:
            rawData = rawData.sort_values(by = 'click', ascending = False)
        if index_end == -1 or index_end > rawData.shape[0]:
            inData = pd.DataFrame(enc.transform(rawData.iloc[:,:-1]).toarray())
            outData = rawData.iloc[:,-1].values
        else:
            inData = pd.DataFrame(enc.transform(rawData.iloc[index_start:index_end,:-1]).toarray())
            outData = rawData.iloc[index_start:index_end,-1].values
    else:
        if not(os.path.isfile(encoder_name)):
            enc = OneHotEncoder()
            enc.fit(rawData)
            pickle.dump(enc, open(encoder_name, "wb"))
        else:
            enc = pickle.load(open(encoder_name,"rb"))
        if index_end == -1 or index_end > rawData.shape[0]:
            inData = pd.DataFrame(enc.transform(rawData).toarray())
            outData = pd.DataFrame([0]*rawData.shape[0]).T.values[0]
        else:
            inData = pd.DataFrame(enc.transform(rawData.iloc[index_start:index_end,:]).toarray())
            outData = pd.DataFrame([0]*(index_end - index_start)).T.values[0]

    return inData, outData
        
def convert_DataToCsv(data, name):
    """Save the DataFrame with the associated name."""
    pd.DataFrame(data).to_csv(name, index=False, header=False)
