from NeuralNetworkModel import *
from ReadCSV import *
import pandas as pd
import time

data_size = [3100]
n_layers = [100]
ratio_p = [0.000333 * i for i in range(1,10)]
k_samples = [5,7,10,11,14,16,20]
activation = ['tanh']
solver = ['lbfgs']
seed = [None]

df = pd.DataFrame()
n_combination = len(data_size) * len(n_layers) * len(ratio_p) * len(k_samples) * len(activation) *  len(solver) * len(seed)
i = 0
iStart = 0
name = "Results.csv"

for d in data_size:
    for n in n_layers:
        for r in ratio_p:
            for k in k_samples:
                for a in activation:
                    for s in solver:
                        for se in seed:
                            i = i+1
                            if i < iStart:
                                continue
                            print i,"/",n_combination,":",d,n,r,k,a,s,se
                            stTime = time.time()
                            F_values = NN_Model(d,n,r,k,a,s,se)
                            fnTime = time.time()-stTime
                            df = df.append([[d,n,r,k,a,s,se,F_values[0],F_values[1],fnTime]],True)
                            convert_DataToCsv(df,name)

                            
df.columns = ['Data Size', 'Number of layers', 'Ratio of 1 over all',
              'Number of k-samples', 'Type of activation', 'Type of solver',
              'Seed use', 'F-Measure mean', 'Last F-Measure', 'Time']
convert_DataToCsv(df,name)

