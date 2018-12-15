from ReadCSV import *
#import pickle
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
import random as rd
import numpy as np

def Results(y_true, y_pred, print_res = True):
    F = f1_score(y_true, y_pred)
    if print_res:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print "Accuracy:",acc,"\nPrecision:",prec,"\nRecall:",rec,"\nF-Measure:",F,"\n"
    return F

def NN_Model(data_size, n_layers, ratio_p, k, activation, solver, seed):
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio_p, size = data_size)
    print "Data extraction completed\n"

    #Brain
    model = nn.MLPClassifier(hidden_layer_sizes = n_layers, activation = activation,
                         solver = solver, random_state = seed)

    #k-fold alt-model
    F_measure = 0.0
    k_size = X.shape[0] / k
    for i in range(100):
        k_test = rd.randint(0, k) * k_size
        for j in range(0, k*k_size, k_size):
            if j == k_test:
                continue
            sampleBot, sampleTop = j, j + k_size
            model.fit(X[sampleBot:sampleTop],y[sampleBot:sampleTop])
        F_i = Results(y[k_test:k_test+k_size],model.predict(X[k_test:k_test+k_size]),False)
        F_measure = F_measure + F_i

    return F_measure/100, F_i

