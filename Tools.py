# Documentation:    https://www.kaggle.com/rafjaa/resampling-strategies-for-imbalanced-datasets
#                   https://beckernick.github.io/oversampling-modeling/

from Sample_gen import *

from imblearn.under_sampling import TomekLinks

from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import VotingClassifier

from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score

import gc
gc.collect()

# Samplers
tomek = TomekLinks(sampling_strategy = 'auto' ,n_jobs = 4)
print "Sample tools ready"


# Methods
names = ["DT_2es","RF_2es10","RF_2es30","DT_2el","RF_2el10","RF_2el30","DT_2gs","RF_2gs10","RF_2gs30","DT_2gl","RF_2gl10","RF_2gl30",
         "DT_4es","RF_4es10","RF_4es30","DT_4el","RF_4el10","RF_4el30","DT_4gs","RF_4gs10","RF_4gs30","DT_4gl","RF_4gl10","RF_4gl30",
         "DT_9es","RF_9es10","RF_9es30","DT_9el","RF_9el10","RF_9el30","DT_9gs","RF_9gs10","RF_9gs30","DT_9gl","RF_9gl10","RF_9gl30",
         "AB_2S","AB_2SR","AB_4S","AB_4SR","AB_9S","AB_9SR"]

def generate_classifiers():
    clf = []
    
    # Creating Decision Tree + Random Forest + Ada Boost
    clf = clf + [DecisionTreeClassifier(max_depth = 4, criterion = "gini", max_features = "sqrt")]
    clf = clf + [DecisionTreeClassifier(max_depth = 9, criterion = "entropy", max_features = "sqrt")]
    clf = clf + [DecisionTreeClassifier(max_depth = 9, criterion = "entropy", max_features = "log2")]
    clf = clf + [DecisionTreeClassifier(max_depth = 9, criterion = "gini", max_features = "sqrt")]
    clf = clf + [DecisionTreeClassifier(max_depth = 9, criterion = "gini", max_features = "log2")]

    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 4, criterion = "gini", n_jobs = 1)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 9, criterion = "entropy", n_jobs = 1)]
    clf = clf + [RandomForestClassifier(n_estimators = 30, max_depth = 9, criterion = "entropy", n_jobs = 1)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 9, criterion = "entropy", n_jobs = 1)]
    clf = clf + [RandomForestClassifier(n_estimators = 10, max_depth = 9, criterion = "gini", n_jobs = 1)]
    clf = clf + [RandomForestClassifier(n_estimators = 30, max_depth = 9, criterion = "gini", n_jobs = 1)]

    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth = 2, criterion = "entropy"),learning_rate = 1.5,algorithm = "SAMME")]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, criterion = "entropy"),learning_rate = 1.5,algorithm = "SAMME")]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth = 4, criterion = "entropy"),learning_rate = 1.5,algorithm = "SAMME.R")]
    clf = clf + [AdaBoostClassifier(DecisionTreeClassifier(max_depth = 9, criterion = "entropy"),learning_rate = 1.5,algorithm = "SAMME")]

    return zip(names[:-1], clf)


# Evaluation
def Results(y_true, y_pred, print_res = True):
    F = f1_score(y_true, y_pred)
    if print_res:
        prec = precision_score(y_true, y_pred)
        rec = recall_score(y_true, y_pred)
        acc = accuracy_score(y_true, y_pred)
        print "Accuracy:",acc,"\nPrecision:",prec,"\nRecall:",rec,"\nF-Measure:",F,"\n"
    return F


# Statistics
names_F15_freq = [0] * len(names)
with open('Stats.txt', 'a') as f:
    f.write("\n\n\n\nNew Generation of methods above f-score of 0.15\n")
    for item in names:
        f.write("%s," % item)
    f.write("\n")
    for item in names_F15_freq:
        f.write("%s," % item)


# Output / Prediction
def generate_output(model,name = "output.csv",name_model = "model"):
    pickle.dump(clf, open(name_model, "wb"))
    y_out = model.predict(X_out)
    convert_DataToCsv(y_out,name)
