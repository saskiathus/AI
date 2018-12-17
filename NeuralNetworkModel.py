from ReadCSV import *
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

def Results(y_true, y_pred, print_res = True):
    F = f1_score(y_true, y_pred)
    if print_res:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print "Accuracy:",acc,"\nPrecision:",prec,"\nRecall:",rec,"\nF-Measure:",F,"\n"
    return F

def NN_Model(m_i,sol,act,alpha,tol,ratio,n_lays,F_min = 0):
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = 5000)
    X_train, y_train = convert_CsvToData('TraData.csv', limit = 300000) #Ratio = 0.0005
    print "Data extraction completed"

    #Brain
    model = nn.MLPClassifier(hidden_layer_sizes = n_lays,activation = act, tol = tol,
                             alpha = alpha, solver = sol, max_iter = m_i, random_state = 42)

    #Prediction
    model = model.fit(X,y)
    Fm = Results(y_train,model.predict(X_train))

    #Real Output
    if Fm > F_min:
        print "Start Real Output!\n"
        X_test, y_test = convert_CsvToData('input.csv', TrainingData = False)
        y_test = model.predict(X_test)
        convert_DataToCsv(y_test,'output.csv')
        
    return Fm
