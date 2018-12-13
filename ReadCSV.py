#https://scikit-learn.org/stable/datasets/index.html

#Use pandas to avoid the commas issue
import pandas as pd
import random as rd
from sklearn.utils import shuffle

def convert_CsvToData(name, TrainingData = True):
    #Extract data from csv file, and put Nan value to 0
    rawData = pd.read_csv(name, delimiter = ',', dtype={"dclkVerticals": object})
    rawData = rawData.fillna(0)

    #Neural Networks doesn't support string, so convert them
    for i in range(10):
        rawData[rawData.columns[i]] = rawData[rawData.columns[i]].apply(convert_string)

    #Seperate inputs and outputs
    if TrainingData:    
        inData = rawData.iloc[:,:-1]
        outData = rawData.iloc[:,-1].values
    else:
        inData = rawData
        outData = pd.DataFrame([0]*rawData.shape[0]).T.values[0]
    return (inData,outData)

def convert_CsvToData_ratio(name, ratio = 0.5, size = 10000):
    #ratio is the ratio of positif over the total
    data = convert_CsvToData(name, False)[0]
    data = data.sort_values(by = 'click')
    n_d = data.shape[0]
    data.index = [i for i in range(n_d)]
    i_cut = data['click'].idxmax()
    X = pd.DataFrame(data[0:1])
    r = (1.0,0.0)
    for i in range(size):
        if r[1]/(r[1]+r[0]) < ratio:
            i_rd = rd.randint(i_cut,n_d-1)
            r = (r[0],r[1]+1)
        else:
            i_rd = rd.randint(0,i_cut-1)
            r = (r[0]+1,r[1])
        X = X.append(data[i_rd:i_rd+1], ignore_index = True)
        #print i_rd, data[i_rd:(i_rd+1)]
    X = shuffle(X)
    return (X.iloc[:,:-1],X.iloc[:,-1].values)
        
def convert_DataToCsv(data, name):
    pd.DataFrame(data).to_csv(name, index=False)
    
def convert_string(string):
    """Convert string into float by using Unicode decimal"""
    if type(string) != str:
        return float(string)
    
    value = 0
    for c in string:
        value = value + ord(c)
    return float(value)


#Examples
##(X,y) = convert_CsvToData('TraData.csv')
##(Xt, yt) = convert_CsvToData('input.csv',TrainingData = False)
##convert_DataToCsv(yt,'output.csv')
