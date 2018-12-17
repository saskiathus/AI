from ReadCSV import *
import sklearn.neural_network as nn
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
# NEW Traning methods library
# tree
from sklearn import tree
#Random forest
from sklearn.ensemble import RandomForestClassifier 
#ensemble
from sklearn.ensemble import VotingClassifier 




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


# NEW Traning methods -> Tree
def Tree_Model():
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = 5000)
    X_train, y_train = convert_CsvToData('TraData.csv', limit = 300000) #Ratio = 0.0005
    print "Data extraction completed"

    #Brain #//train the model
    model_Tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)	 #//select model parameters
    					                     

    
    #Prediction and fitting
    model_Tree = model_Tree.fit(X,y)
    Fm = Results(y_train,model_Tree.predict(X_train))
    

    #Real Output
    if Fm > F_min:
        print "Start Real Output!\n"
        X_test, y_test = convert_CsvToData('input.csv', TrainingData = False)
        y_test = model.predict(X_test)
        convert_DataToCsv(y_test,'output.csv')
        
    return Fm


# NEW Traning methods -> Random forest
def RandomForest_Model():
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = 5000)
    X_train, y_train = convert_CsvToData('TraData.csv', limit = 300000) #Ratio = 0.0005
    print "Data extraction completed"

    #Brain #//train the model
    model_RF = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=0, n_jobs=2, max_depth=6, max_features='log2')

    #Prediction and fitting
    model_RF = model_RF.fit(X,y)
    Fm = Results(y_train,model_RF.predict(X_train))
    
    #Real Output
    if Fm > F_min:
        print "Start Real Output!\n"
        X_test, y_test = convert_CsvToData('input.csv', TrainingData = False)
        y_test = model.predict(X_test)
        convert_DataToCsv(y_test,'output.csv')
        
    return Fm

def Ensemble():
    #Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = 5000)
    X_train, y_train = convert_CsvToData('TraData.csv', limit = 300000) #Ratio = 0.0005
    print "Data extraction completed"
    
    #BrainS
    model_NL = nn.MLPClassifier(hidden_layer_sizes = n_lays,activation = act, tol = tol,
                             alpha = alpha, solver = sol, max_iter = m_i, random_state = 42)

    model_Tree = tree.DecisionTreeClassifier(criterion='entropy', max_depth=7)	 
    model_RF = RandomForestClassifier(criterion='entropy', n_estimators=500, random_state=0, n_jobs=2, max_depth=6, max_features='log2')
    
    
    #Conbined Brains
    
    model_E = VotingClassifier(estimators=[('NL', model_NL), ('Tree', model_Tree), ('rf', model_RF)], voting='hard', weights=[2,1,1])
    #voting classifier is the ensemble model
    # estimators are the model we used to fit the data e.g. NL, RF, Tree
    #Parameters are 
    #              voting = hard/soft
    #              weights = [number1 ,number2, number3 ]
    
    #fitting and predicting
    model_E = model_E.fit(X,y)
    Fm = Results(y_train,model_E.predict(X_train))
    
    
    #Real Output
    if Fm > F_min:
        print "Start Real Output!\n"
        X_test, y_test = convert_CsvToData('input.csv', TrainingData = False)
        y_test = model.predict(X_test)
        convert_DataToCsv(y_test,'output.csv')
        
    return Fm
    
    
    

    
    
    
    
    
    
    
    