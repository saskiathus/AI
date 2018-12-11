#https://scikit-learn.org/stable/datasets/index.html

#Use pandas to avoid the commas issue
import pandas as pd

#Extract data from csv file, and put Nan value to 0
data = pd.read_csv('TraData.csv',delimiter = ',')
data = data.fillna(0)

#Convert string into float by using Unicode decimal
def convert_string(string):
    if type(string) != str:
        return float(string)
    
    value = 0
    for c in string:
        value = value + ord(c)
    return float(value)

for i in range(10):
    data[data.columns[i]] = data[data.columns[i]].apply(convert_string)

#Seperate inputs and outputs
X = data.iloc[:,:-1]
y = data.iloc[:,-1].values



#Neural Networks
import sklearn.neural_network as nn

#Number of layers
n = 10
model = nn.MLPClassifier(n)
model.fit(X,y)

#Example: model.predict(data.iloc[2020:2030,:-1])
