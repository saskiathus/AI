from ReadCSV import *
(X,y) = convert_CsvToData('TraData.csv')
(Xt, yt) = convert_CsvToData('input.csv',TrainingData = False)

#Neural Networks library
import sklearn.neural_network as nn

#Number of layers
n = 10
model = nn.MLPClassifier(n)
model.fit(X,y)

yt = model.predict(Xt)
convert_DataToCsv(yt,'output.csv')
