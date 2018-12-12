from ReadCSV import *
(X,y) = convert_CsvToData('TraData.csv')
(Xt, yt) = convert_CsvToData('input.csv',TrainingData = False)
print "Data extraction completed\n"

#Neural Networks library
import sklearn.neural_network as nn

#Number of layers
n = 20
model = nn.MLPClassifier(n)

#k-fold model
import random as rd
import numpy as np

for i in range(10):
    k = rd.randint(5,15)
    k_size = float(X.shape[0])/k
    for j in range(k - 1):
        sampleBot = int(k_size * j)
        sampleUp = int(k_size * (j+1))
        model.fit(X[sampleBot:sampleUp],y[sampleBot:sampleUp])
    print "Score:",model.score(X[sampleUp:],y[sampleUp:])


yt = model.predict(Xt)
convert_DataToCsv(yt,'output.csv')
