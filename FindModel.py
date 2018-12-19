from ReadCSV import *
import pickle
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.neural_network import MLPClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

##Evaluation
def Results(y_true, y_pred, print_res = True):
    F = f1_score(y_true, y_pred)
    if print_res:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print "Accuracy:",acc,"\nPrecision:",prec,"\nRecall:",rec,"\nF-Measure:",F,"\n"
    return F
Fmax, Fsup, Fmoy = 0.0, 0.0, 0.0
F_15 = []


## Methods
names = ["DT_5e","DT_5","DT_e","DT","ADC_Tla","ADC_a","MLP","ENS"]

def generate_classifiers():
    clf = []
    clf = clf + [DecisionTreeClassifier(max_depth=5,criterion="entropy")]
    clf = clf + [DecisionTreeClassifier(max_depth=5)]
    clf = clf + [DecisionTreeClassifier(criterion="entropy")]
    clf = clf + [DecisionTreeClassifier()]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth=5,criterion="entropy"),learning_rate = 1.5,algorithm="SAMME")]
    clf = clf + [AdaBoostClassifier(algorithm="SAMME")]
    clf = clf + [MLPClassifier(hidden_layer_sizes = (500,450,400),activation = 'tanh', tol = 1e-7, warm_start = True,
                               alpha = 2e-5, solver = 'adam', max_iter = 700, random_state = None)]

    mod = []
    for name, clf in zip(names[:-1], clf):
        mod = mod + [(name,clf)]
    mod = mod + [("ENS",VotingClassifier(estimators=mod))]
    return mod

## Data
X_train, y_train = convert_CsvToData('TraData.csv', index_end = 300000) #Ratio = 0.0005.
X_trainb, y_trainb = convert_CsvToData('TraData.csv', index_start = 600000, index_end = 900000) #Ratio = 0.0005.
X_out,y_out = convert_CsvToData('input.csv', TrainingData = False)
print "Data extraction completed"

def generate_output(model,name = "output.csv",name_model = "model"):
    pickle.dump(clf, open(name_model, "wb"))
    y_out = model.predict(X_out)
    convert_DataToCsv(y_out,name)


## Parameters
training_size = 3000
ratio = random.random()*5e-3
len_F15_max = 15


while Fmax < 0.9:
    # Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = training_size)
    print "Data extraction completed"

    classifiers = generate_classifiers()
    for name, clf in classifiers:
        print "\nName:",name
        # Learning
        clf.fit(X, y)

        # Prediction
        print "Prediction..."
        Fm = Results(y_train,clf.predict(X_train))

        # Display Results
        print "Size = %s, Ratio = %.6f\nFm = %.5f and Fsup = %.5f\n" % (training_size, ratio, Fm, Fsup)

        # Comparing Results
        if Fm > Fsup:
            Fsup = Fm
            if Fm > Fmax:
                Fmax = Fm
                generate_output(clf)
            if Fm > 0.15:
                F_15 = F_15 + [(name,clf)]
        Fmoy = Fmoy + Fm

    # Changing the threshold of F-measurement. Good Fm on large dataset are more intersting.    
    Fsup = (99 * Fsup + Fmoy / len(classifiers))/100
    Fmoy = 0.0

    # Testing the best results
    print "\nLength of F_15:",len(F_15)
    if len(F_15) > len_F15_max:
        print "Starting process to determine best method!"
        results = []
        for name, clf in F_15:
            results = results.append(Results(y_trainb,clf.predict(X_trainb),False))
        F_15 = [x for _,x in sorted(zip(results,F_15),reverse = True)]
        F_15 = F_15[:5]
        best_Fm = Results(y_trainb,F_15[0][1].predict(X_trainb),False)
        saveName = "_%s_%.4f.csv" % (F_15[0][0],best_Fm)
        generate_output(F_15[0][1],"output"+saveName,"model"+saveName)
        print "Best model %s save with Fm = %.4f" % (F_15[0][0],best_Fm)

        est = []
        for name, clf in F_15:
            est = est + [(name,clf)]
        estimator = VotingClassifier(estimators=est)
        print "Comparison with estimator: Fm = ", Results(y_trainb,estimator.predict(X_trainb), False)

    # Preparing next step
    print "Next loop in Preparation...\n\n\n"
    training_size = training_size + 200
    ratio = random.random()*5e-3
    while ratio < 5e-4:
        ratio = random.random()*5e-3


####### Classifiers tested #######
##from sklearn.neural_network import MLPClassifier
##from sklearn.neighbors import KNeighborsClassifier
##from sklearn.svm import SVC
##from sklearn.gaussian_process import GaussianProcessClassifier
##from sklearn.gaussian_process.kernels import RBF
##from sklearn.tree import DecisionTreeClassifier
##from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
##from sklearn.naive_bayes import GaussianNB
##from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
##
##names = ["Random Forest", "Naive Bayes", "Decision Tree", "Linear SVM",
##         "QDA", "AdaBoost", "RBF SVM", "Nearest Neighbors", "Neural Net",
##         "Gaussian Process"]
##
##classifiers = [
##    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
##    GaussianNB(),
##    DecisionTreeClassifier(max_depth=5),
##    SVC(kernel="linear", C=0.025),
##    QuadraticDiscriminantAnalysis(),
##    AdaBoostClassifier(),
##    SVC(gamma=2, C=1),
##    KNeighborsClassifier(3),
##    MLPClassifier(hidden_layer_sizes = (500,500,400),activation = 'tanh', tol = 1e-8, warm_start = True,alpha = 2e-5, solver = 'adam', max_iter = 1000, random_state = None),
##    GaussianProcessClassifier(1.0 * RBF(1.0))]
##
####### Results of F-measure and process time on a small dataset #######
##results = [
##    (0.0000,2),
##    (0.0013,16),
##    (0.0994,3),
##    (0.0000,43),
##    (0.0024,30),
##    (0.0906,16),
##    (0.0791,382),
##    (0.0945,1345),
##    (0.0921,118),
##    (NA,inf)] # Results for Gaussian Process is above 1800 (30min)

