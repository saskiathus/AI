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

def NN_Model(data_size, n_layers, ratio_p, k, activation = 'tanh', solver = 'lbfgs', seed = None, name_brain = 'brain'):
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio_p, size = data_size)
    #Xt, yt = convert_CsvToData('input.csv',TrainingData = False)
    print "Data extraction completed\n"

    #Brain
    name = name_brain
    if not(os.path.isfile(name)):
        model = nn.MLPClassifier(hidden_layer_sizes = n_layers, activation = activation,
                                 solver = solver, random_state = seed)
        pickle.dump(model, open(name, "wb"))
    else:
        model = pickle.load(open(name,"rb"))

    #k-fold alt-model
    F_measure = 0.0
    k_size = X.shape[0] / k
    for i in range(20):
        k_test = rd.randint(0, k-1) * k_size
        for j in range(0, k*k_size, k_size):
            if j == k_test:
                continue
            sampleBot, sampleTop = j, j + k_size
            model = model.fit(X[sampleBot:sampleTop],y[sampleBot:sampleTop])
        F_i = Results(y[k_test:k_test+k_size],model.predict(X[k_test:k_test+k_size]),False)
        F_measure = F_measure + F_i

    pickle.dump(model, open(name, "wb"))
    #yt = model.predict(Xt)
    #convert_DataToCsv(yt,'output.csv')
    return F_measure/100, F_i

