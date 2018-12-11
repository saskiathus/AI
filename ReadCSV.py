#https://scikit-learn.org/stable/datasets/index.html

#Use pandas to avoid the commas issue
import pandas as pd

#Extract data from csv file, and put Nan value to 0
data = pd.read_csv('TraData.csv',delimiter = ',')
data = data.fillna(0)

#Seperate inputs and outputs
X = data.iloc[:,0:-1]
y = data.iloc[:,-1:]




#Neural Networks
import sklearn.neural_network as nn

#Number of layers
n = 10
model = nn.MLPClassifier(n)
model.fit(X,y)

#Error due to value types, they should be float. Next step: convert string into float
