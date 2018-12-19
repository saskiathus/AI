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

def convert_CsvToData(name, TrainingData = True, index_start = 0, index_end = -1):
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

def convert_CsvToData_ratio(name, ratio = 0.5, size = 10000):
    """Give a sample of data with a ratio of positive/(positive+negative).
        To construct a new set of data, some of the previous can be ignored or duplicated.

    Parameters
    ----------
    name: Name or relative path of the file with the extension.

    ratio: Ratio of positive Target over the total amount of Target.

    size: Size of the generated data.

    Returns
    -------
    inData, outData: tuple containing a DataFrame of Inputs of the datafile and
        an array of Target.
    """
    #Extract data from csv file, and put Nan value to 0
    rawData = pd.read_csv(name, delimiter = ',', dtype={"dclkVerticals": object})
    rawData.drop('ip', axis=1, inplace=True)
    rawData = rawData.fillna(0)

    #Encoder
    if not(os.path.isfile(encoder_name)):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(rawData.iloc[:,:-1])
        pickle.dump(enc, open(encoder_name, "wb"))
    else:
        enc = pickle.load(open(encoder_name,"rb"))
        
    #Rearranging index after sorting and get the index of the first 1.
    n_d = rawData.shape[0]
    rawData = rawData.sort_values(by = 'click')
    rawData.index = [i for i in range(n_d)]
    i_cut = rawData['click'].idxmax()
    data = pd.DataFrame(rawData[:1])
    p, n = 0.0, 1.0
    
    for i in range(size):
        if p/(p+n) < ratio:
            i_rd = rd.randint(i_cut,n_d - 1)
            p = p+1
        else:
            i_rd = rd.randint(0,i_cut - 1)
            n = n+1
        data = data.append(rawData[i_rd:i_rd+1], ignore_index = True)
    data = shuffle(data)
        
    #Encode data
    inData = pd.DataFrame(enc.transform(data.iloc[:,:-1]).toarray())
    outData = data.iloc[:,-1].values
    return inData, outData
        
def convert_DataToCsv(data, name):
    """Save the DataFrame with the associated name."""
    pd.DataFrame(data).to_csv(name, index=False, header=False)
