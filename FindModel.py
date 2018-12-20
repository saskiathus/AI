from Tools import *

# Data
X_train, y_train = convert_CsvToData('TraData.csv', index_end = 200000) #Ratio = 0.0005
X_train_F15, y_train_F15 = convert_CsvToData('TraData.csv', index_start = 600000, index_end = 900000) #Ratio = 0.0005
X_out,y_out = convert_CsvToData('input.csv', TrainingData = False)
print "Data extraction completed\n"

# Initial state
training_size = 1000
ratio = 0.01
n_ones_min = 20
Fmax = 0.0
F_15 = []
len_F15_max = 15
F15_threshold = 0.05


# Testing best results
def F15_process(F_15, X, y):
    # Write Statistics
    with open('Stats.txt', 'a') as f:
        f.write("\n")
        for item in names_F15_freq:
            f.write("%s," % item)

    # Evaluating best models on another set of Data
    results, names_methods = [], ""
    for name, clf in F_15:
        results = results + [Results(y_train_F15,clf.predict(X_train_F15),False)]
        names_methods = names_methods + " " + name
        
    # Sorting best methods by f-score and keep the best 5
    print "List of methods: " + names_methods
    F_15 = [x for _,x in sorted(zip(results,F_15),reverse = True)]
    F_15 = F_15[:5]
    F15_threshold = Results(y_train_F15,F_15[-1][1].predict(X_train_F15),False)

    # Comparing first model and Ensemble of 5, and generate the output
    print "Best model: %s" % F_15[0][0]
    best_Fm = Results(y_train_F15,F_15[0][1].predict(X_train_F15))
    saveName = "_%s_%.4f" % (F_15[0][0],best_Fm)
    
    ens = VotingClassifier(estimators = F_15)
    ens.fit (X, y)
    ens_Fm = Results(y_train,ens.predict(X_train), False)
    if ens_Fm > F15_threshold:
        print "Ensemble: "
        ens_Fm = Results(y_train_F15,ens.predict(X_train_F15))
        if ens_Fm > best_Fm:
            saveName = "_%s_%.4f_%s" % ("ENS",ens_Fm,training_size)
            
    generate_output(F_15[0][1],"output"+saveName+".csv","model"+saveName+".model")

while Fmax < 0.95:
    # Create new sample of Data
    X, y = convert_CsvToData_ratio('TraData.csv', ratio = ratio, size = training_size)
    X, y = tomek.fit_resample(X,y)
    print "Size = %s // Ratio = %.4f // Fmax = %.3f // F_15 = %s // F15_min = %.3f" % (training_size, ratio, Fmax, len(F_15), F15_threshold)

    # Create new classifiers and evaluate them
    classifiers = generate_classifiers()
    for i in range (len(classifiers)):
        name, clf = classifiers[i]
        
        clf.fit(X, y)
        Fm = Results(y_train,clf.predict(X_train), False)

        # Comparing Results
        if Fm > Fmax:
            Fmax = Fm
        if Fm > F15_threshold:
            F_15 = F_15 + [(name,clf)]
            names_F15_freq[i] = names_F15_freq[i] + 1

    # Testing the best results
    if len(F_15) > len_F15_max:
        F15_process(F_15, X, y)

    # Preparing next step
    training_size, ratio = training_size + 250, rd.random()*0.01
    while ratio * training_size < n_ones_min:
        ratio = ratio + 0.0001
    gc.collect()


