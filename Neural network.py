from ReadCSV import *
(X,y) = convert_CsvToData('TraData.csv')
(Xt, yt) = convert_CsvToData('input.csv',TrainingData = False)
print "Data extraction completed\n"

#Neural Networks library
import sklearn.neural_network as nn

#Number of layers
n = 20
model = nn.MLPClassifier(n)

import random as rd
import numpy as np

for i in range(10):
    k = rd.randint(5,15)
    k_size = float(X.shape[0])/k
    for j in range(k - 1):
        sampleBot = int(k_size * j)
        sampleUp = int(k_size * (j+1))
        model.fit(X[sampleBot:sampleUp],y[sampleBot:sampleUp])
    yRes = model.predict(X[sampleUp:])
    diff = yRes - y[sampleUp:]
    print "Number of differences between Training and Test:",np.count_nonzero(diff)
    
#model.fit(X[:100],y[:100])

yt = model.predict(Xt)
convert_DataToCsv(yt,'output.csv')
