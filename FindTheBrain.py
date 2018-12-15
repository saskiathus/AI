from NeuralNetworkModel import *
from ReadCSV import *
import pandas as pd
import time

data_size = [100,1000,5000]
n_layers = [10,20,42,60,100]
ratio_p = [0.6,0.3,0.1,0.01,0.001]
k_samples = [5,7,10,11,14,16,20]
activation = ['logistic', 'tanh']
solver = ['lbfgs', 'sgd', 'adam']
seed = [0,None]

df = pd.DataFrame()
n_combination = len(data_size) * len(n_layers) * len(ratio_p) * len(k_samples) * len(activation) *  len(solver) * len(seed)
i = 0
comb = []

for d in data_size:
    for n in n_layers:
        for r in ratio_p:
            for k in k_samples:
                for a in activation:
                    for s in solver:
                        for se in seed:
                            comb.append([d,n,r,k,a,s,se])
                            i = i+1
                            if i < 581:
                                continue
                            print i,"/",n_combination,":",d,n,r,k,a,s,se
                            stTime = time.time()
                            F_values = NN_Model(d,n,r,k,a,s,se)
                            fnTime = time.time()-stTime
                            df = df.append([[d,n,r,k,a,s,se,F_values[0],F_values[1],fnTime]],True)
                            convert_DataToCsv(df,"Results.csv")

                            
df.columns = ['Data Size', 'Number of layers', 'Ratio of 1 over all',
              'Number of k-samples', 'Type of activation', 'Type of solver',
              'Seed use', 'F-Measure mean', 'Last F-Measure', 'Time']
convert_DataToCsv(df,"Results.csv")

