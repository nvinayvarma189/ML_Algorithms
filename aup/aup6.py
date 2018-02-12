import pandas
import numpy
from sklearn import neural_network,cross_validation
from sklearn import svm

data=pandas.read_csv(r'G:\AI\breast-cancer-wisconsin.data.txt')

data.drop(['id'],1,inplace=True)

data.replace('?',-9999,inplace=True)
x=numpy.array(data.drop(['class'],1))
y=numpy.array(data['class'])

xtrain,xtest,ytrain,ytest=cross_validation.train_test_split(x,y,test_size=0.2)


alg1=neural_network.MLPClassifier()
alg1.fit(xtrain,ytrain)
accuracy=alg1.score(xtest,ytest)
print(accuracy)


alg2=svm.SVC()
alg2.fit(xtrain,ytrain)
accuracy=alg2.score(xtest,ytest)
print(accuracy)








