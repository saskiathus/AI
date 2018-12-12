#https://scikit-learn.org/stable/datasets/index.html

#Use pandas to avoid the commas issue
import pandas as pd

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

def convert_DataToCsv(data, name):
    pd.DataFrame(data).to_csv(name, index=False, header = ["click"])
    
def convert_string(string):
    """Convert string into float by using Unicode decimal"""
    if type(string) != str:
        return float(string)
    
    value = 0
    for c in string:
        value = value + ord(c)
    return float(value)


#Examples
(X,y) = convert_CsvToData('TraData.csv')
(Xt, yt) = convert_CsvToData('input.csv',TrainingData = False)
##convert_DataToCsv(yt,'output.csv')



#Neural Networks
import sklearn.neural_network as nn

#Number of layers
n = 10
model = nn.MLPClassifier(n)
model.fit(X,y)

yt = model.predict(Xt)
convert_DataToCsv(yt,'output.csv')
