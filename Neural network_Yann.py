from ReadCSV import *
import pickle
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
import random as rd
import numpy as np

def Results(brain, y_true, y_pred):
    #pickle.dump(brain, open("brain", "wb"))
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    F = 2 * prec * rec / (prec + rec)
    print "Accuracy:",acc,"\nPrecision:",prec,"\nRecall:",rec,"\nF-Measure:",F,"\n"
    return F

#X, y = convert_CsvToData('TraData.csv')
X, y = convert_CsvToData_ratio('TraData.csv')
Xt, yt = convert_CsvToData('input.csv',TrainingData = False)
print "Data extraction completed\n"

#Number of layers
n = 100
model = nn.MLPClassifier(hidden_layer_sizes = n, activation = 'logistic',
                         solver = 'lbfgs',random_state = None)

#k-fold model
for i in range(1000):
    k = rd.randint(5,20)
    k_step = X.shape[0]/k
    for j in range(0, (k-1)*k_step, k_step):
        sampleBot, sampleTop = j, j + k_step
        model.fit(X[sampleBot:sampleTop],y[sampleBot:sampleTop])
    print i+1,"."
    Results(model,y[sampleTop:],model.predict(X[sampleTop:]))
    yt = model.predict(Xt)
    convert_DataToCsv(yt,'output.csv')

