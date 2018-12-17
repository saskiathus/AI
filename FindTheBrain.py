from NeuralNetworkModel import *
from ReadCSV import *
import pandas as pd
import time
import random
from math import log10
from math import ceil

basic_100 = (100,100,100,100)
basic_200 = (200,200,200)
n_lays = [basic_200+basic_100]
ratio = [0.005,0.004]
tol = [1e-4,5e-5]
alpha = [2e-5,9e-6]
activation = ['tanh']
solver = ['adam']
max_iter = [700,1500]


name = "Results_Occam_V5.csv"
df = pd.DataFrame()
n_combi = 3*len(n_lays)*len(ratio)*len(tol)*len(alpha)*len(activation)*len(solver)*len(max_iter)
i = 0
F_max = 0

for m in max_iter:
    for s in solver:
        for a in activation:
            for af in alpha:
                for t in tol:
                    for r in ratio:
                        for n in n_lays:
                            for k in range(3):
                                i = i+1
                                print i,"/",n_combi,":",m,s,a,af,t,r,n,F_max
                                Fm = NN_Model(m,s,a,af,t,r,n,F_max)
                                if Fm > F_max:
                                    F_max = Fm
                                print "F_measure = %.4f\n" % Fm
                                df = df.append([[m,s,a,af,t,r,n,Fm]],True)
                                convert_DataToCsv(df,name)
  
df.columns = ['Data Size', 'Number of layers', 'Ratio of 1 over all',
              'Number of k-samples', 'Type of activation', 'Type of solver',
              'Seed use', 'F-Measure mean', 'Last F-Measure', 'Time']
convert_DataToCsv(df,name)

