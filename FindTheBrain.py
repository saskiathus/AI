from NeuralNetworkModel import *
from ReadCSV import *
import pandas as pd
import time
import random
from math import log10
from math import ceil

d, n = 1000, 100 #data_size, n_layers
#k_samples = [5,7,10,11,14,16,20]
a, s, se = 'tanh', 'lbfgs', None #activation, solver, seed

name = "Results_10lays.csv"
na = 'brain_10'
df = pd.DataFrame()
i = 0

ratio_p = 0.5 #ratio_p
F_measure_Goal = 0.6
ratio_num = log10(F_measure_Goal)
ratio_denom = F_measure_Goal + log10(1.0 + 1.0/F_measure_Goal)

while(ratio_p > 0.01):
    k = random.randint(4,20)
    d = int(ceil(20.0*k/ratio_p))
    i = i+1
    print i,":",d,n,ratio_p,k
    stTime = time.time()
    F_values = NN_Model(d,n,ratio_p,k,a,s,se,name_brain = na)
    fnTime = time.time()-stTime
    print "F_measure = %.4f\n" % F_values[0]
    df = df.append([[d,n,ratio_p,k,a,s,se,F_values[0],F_values[1],fnTime]],True)
    convert_DataToCsv(df,name)

    ratio_p = F_measure_Goal * (ratio_p + log10(1 + F_measure_Goal - F_values[0]) - ratio_num)/ratio_denom
                            
df.columns = ['Data Size', 'Number of layers', 'Ratio of 1 over all',
              'Number of k-samples', 'Type of activation', 'Type of solver',
              'Seed use', 'F-Measure mean', 'Last F-Measure', 'Time']
convert_DataToCsv(df,name)

