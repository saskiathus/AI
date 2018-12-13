### ENCODER ###
import numpy as np
from sklearn.preprocessing import OneHotEncoder

enc = OneHotEncoder()
enc.fit([['',2,'ca'],[2,2,2],['ca','ca','ca']])
print enc.transform([['ca',2,2]]).toarray()


### SAVE ###
import pickle

# store object information
pickle.dump(enc, open("test", "wb"))

# read information from file
enc = pickle.load(open("test","rb"))
