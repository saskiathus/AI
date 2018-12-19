# Documentation:    https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
#                   https://beckernick.github.io/oversampling-modeling/

from ReadCSV import *
import pickle
import random

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import TomekLinks
from imblearn.combine import SMOTETomek

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
names = ["DT_5e","DT_5","DT_2e","DT_2","DT_2eSqrt","DT_2Sqrt","DT_2eLog","DT_2Log","DT_e","DT","RFC_2e","RFC_4e","RFC_2","RFC_4","ADC_T4la","ADC_T2la","ADC_a","ENS"]

def generate_classifiers():
    clf = []
    clf = clf + [DecisionTreeClassifier(max_depth=4,criterion="entropy")]
    clf = clf + [DecisionTreeClassifier(max_depth=4)]
    clf = clf + [DecisionTreeClassifier(max_depth=2,criterion="entropy")]
    clf = clf + [DecisionTreeClassifier(max_depth=2)]
    clf = clf + [DecisionTreeClassifier(max_depth=2,criterion="entropy",max_features="sqrt")]
    clf = clf + [DecisionTreeClassifier(max_depth=2,max_features="sqrt")]
    clf = clf + [DecisionTreeClassifier(max_depth=2,criterion="entropy",max_features="log2")]
    clf = clf + [DecisionTreeClassifier(max_depth=2,max_features="log2")]
    clf = clf + [DecisionTreeClassifier(criterion="entropy")]
    clf = clf + [DecisionTreeClassifier()]
    clf = clf + [RandomForestClassifier(n_estimators = 10, criterion = "entropy", max_depth = 2, n_jobs = 3)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, criterion = "entropy", max_depth = 4, n_jobs = 3)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 2, n_jobs = 3)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 4, n_jobs = 3)]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth=4,criterion="entropy"),learning_rate = 1.5,algorithm="SAMME")]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth=2,criterion="entropy"),learning_rate = 1.5,algorithm="SAMME")]
    clf = clf + [AdaBoostClassifier(algorithm="SAMME")]

    mod = []
    for name, clf in zip(names[:-1], clf):
        mod = mod + [(name,clf)]
    mod = mod + [("ENS",VotingClassifier(estimators=mod))]
    return mod

## Data
X_train, y_train = convert_CsvToData('TraData.csv', index_end = 400000) #Ratio = 0.0005
X_trainb, y_trainb = convert_CsvToData('TraData.csv', index_start = 500000, index_end = 900000) #Ratio = 0.0005
X_out,y_out = convert_CsvToData('input.csv', TrainingData = False)


smote = SMOTE(sampling_strategy = minority, k_neighbors = 3, n_jobs = 3)
tomek = TomekLinks(sampling_strategy = 'auto' ,n_jobs =3)
smt = SMOTETomek(smote = smote, tomek = tomek)
print "Data extraction completed"

def generate_output(model,name = "output.csv",name_model = "model"):
    pickle.dump(clf, open(name_model, "wb"))
    y_out = model.predict(X_out)
    convert_DataToCsv(y_out,name)


## Parameters
training_size = 1000
len_F15_max = 15
F15_threshold = 0.10


while Fmax < 0.9:
    # Data
    X, y = convert_CsvToData('TraData.csv',index_end = training_size, Sort = True)
    X, y = smt.fit_resample(X,y)
    print "Data extraction completed"

    classifiers = generate_classifiers()
    for name, clf in classifiers:
        print "\nName:",name
        # Learning
        clf.fit(X, y)

        # Prediction
        print "Prediction..."
        Fm = Results(y_train,clf.predict(X_train), False)

        # Display Results
        print "Size = %s // Fm = %.5f // Fsup = %.5f\n" % (training_size, Fm, Fsup)

        # Comparing Results
        if Fm > Fsup:
            Fsup = Fm
            if Fm > Fmax:
                Fmax = Fm
                generate_output(clf)
        if Fm > F15_threshold:
            F_15 = F_15 + [(name,clf)]                
        Fmoy = Fmoy + Fm

    # Changing the threshold of F-measurement. Good Fm on large dataset are more intersting.    
    Fsup = (99 * Fsup + Fmoy / len(classifiers))/100
    Fmoy = 0.0

    # Testing the best results
    print "\nLength of F_15:",len(F_15)
    if len(F_15) > len_F15_max:
        print "Starting process to determine best method!"        
        results, names_methods = [], ""
        for name, clf in F_15:
            results = results + [Results(y_trainb,clf.predict(X_trainb),False)]
            names_methods = names_methods + " " + name
        print "List of methods: " + names_methods
        F_15 = [x for _,x in sorted(zip(results,F_15),reverse = True)]
        F_15 = F_15[:5]
        F15_threshold = Results(y_trainb,F_15[4][1].predict(X_trainb))
        print "Best model:
        best_Fm = Results(y_trainb,F_15[0][1].predict(X_trainb))
        saveName = "_%s_%.4f" % (F_15[0][0],best_Fm)
        generate_output(F_15[0][1],"output"+saveName+".csv","model"+saveName+".model")
        print "Best model %s save with Fm = %.4f" % (F_15[0][0],best_Fm)

##        estimator = VotingClassifier(estimators=F_15)
##        res = Results(y_trainb,estimator.predict(X_trainb), False)
##        print "Comparison with estimator: Fm =",res

    # Preparing next step
    print "Next loop in Preparation...\n\n\n"
    training_size = training_size + 200


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

