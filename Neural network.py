from ReadCSV_V2 import *
#X, y = convert_CsvToData('TraData.csv')
X, y = convert_CsvToData_ratio('TraData.csv')
Xt, yt = convert_CsvToData('input.csv',TrainingData = False)
print "Data extraction completed\n"

#Neural Networks library
import sklearn.neural_network as nn

#Number of layers
n = 100
model = nn.MLPClassifier(hidden_layer_sizes = n, activation = 'logistic',
                         solver = 'lbfgs',random_state = None)

#k-fold model
import random as rd
import numpy as np

for i in range(1000):
    k = rd.randint(5,20)
    k_size = float(X.shape[0])/k
    for j in range(k - 1):
        sampleBot = int(k_size * j)
        sampleTop = int(k_size * (j+1))
        model.fit(X[sampleBot:sampleTop],y[sampleBot:sampleTop])
    print i+1,". Score:",model.score(X[sampleTop:],y[sampleTop:])
    yt = model.predict(Xt)
    convert_DataToCsv(yt,'output.csv')
