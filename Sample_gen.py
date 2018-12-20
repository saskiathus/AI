# Documentation : https://scikit-learn.org/stable/datasets/index.html

# Use pandas to avoid the commas' issue
import pandas as pd
import numpy
import random as rd
import os
import pickle
from sklearn.utils import shuffle
from sklearn.preprocessing import OneHotEncoder

# Parameter
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
    # Extract data from csv file, and put Nan value to 0
    rawData = pd.read_csv(name, delimiter = ',', dtype={"dclkVerticals": object})
    rawData.drop('ip', axis=1, inplace=True)
    rawData = rawData.fillna(0)

    # Neural Networks doesn't support string, so convert them => Encoder
    if TrainingData:
        # Create or Load the encoder
        if not(os.path.isfile(encoder_name)):
            enc = OneHotEncoder(handle_unknown='ignore')
            enc.fit(rawData.iloc[:,:-1])
            pickle.dump(enc, open(encoder_name, "wb"))
        else:
            enc = pickle.load(open(encoder_name,"rb"))
        # Sort Data: 1 at the top of the DataFrame
        if Sort:
            rawData = rawData.sort_values(by = 'click', ascending = False)
        # Prepare Data
        if index_end == -1 or index_end > rawData.shape[0]:
            inData = pd.DataFrame(enc.transform(rawData.iloc[:,:-1]).toarray())
            outData = rawData.iloc[:,-1].values
        else:
            inData = pd.DataFrame(enc.transform(rawData.iloc[index_start:index_end,:-1]).toarray())
            outData = rawData.iloc[index_start:index_end,-1].values
    else:
        # Create or Load the encoder
        if not(os.path.isfile(encoder_name)):
            enc = OneHotEncoder()
            enc.fit(rawData)
            pickle.dump(enc, open(encoder_name, "wb"))
        else:
            enc = pickle.load(open(encoder_name,"rb"))
        # Prepare Data
        if index_end == -1 or index_end > rawData.shape[0]:
            inData = pd.DataFrame(enc.transform(rawData).toarray())
            outData = pd.DataFrame([0]*rawData.shape[0]).T.values[0]
        else:
            inData = pd.DataFrame(enc.transform(rawData.iloc[index_start:index_end,:]).toarray())
            outData = pd.DataFrame([0]*(index_end - index_start)).T.values[0]

    return inData, outData

def convert_CsvToData_ratio(name, ratio = 0.005, size = 10000):
    # Extract data from csv file, and put Nan value to 0
    rawData = pd.read_csv(name, delimiter = ',', dtype={"dclkVerticals": object})
    rawData.drop('ip', axis=1, inplace=True)
    rawData = rawData.fillna(0)
    
    # Neural Networks doesn't support string, so convert them => Encoder
    if not(os.path.isfile(encoder_name)):
        enc = OneHotEncoder(handle_unknown='ignore')
        enc.fit(rawData.iloc[:,:-1])
        pickle.dump(enc, open(encoder_name, "wb"))
    else:
        enc = pickle.load(open(encoder_name,"rb"))

    # Sort Data: 1 at the top of the DataFrame
    n_d = rawData.shape[0]
    rawData = rawData.sort_values(by = 'click', ascending = False)
    rawData.index = [i for i in range(n_d)]

    # Create a new DataFrame where data is randomly pick from TraData with a certain ratio
    i_cut = rawData['click'].idxmin() #Position of the first 0  
    n_ones = int(ratio * size) - 1
    data = pd.DataFrame(rawData[:1]) #First row to have a valid DataFrame

    if n_ones < 1:
        n_ones = 0
    for i in range (n_ones):
        i_rd = rd.randint(0, i_cut - 1)
        data = data.append(rawData[i_rd:i_rd+1], ignore_index = True)
    for i in range (size - n_ones):
        i_rd = rd.randint(i_cut, n_d - 1)
        data = data.append(rawData[i_rd:i_rd+1], ignore_index = True)
    data = shuffle(data)

    # Return Data
    inData = pd.DataFrame(enc.transform(data.iloc[:,:-1]).toarray())
    outData = data.iloc[:,-1].values
    return inData, outData
        
def convert_DataToCsv(data, name):
    """Save the DataFrame with the associated name."""
    pd.DataFrame(data).to_csv(name, index=False, header=False)
