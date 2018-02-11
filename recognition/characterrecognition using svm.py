import matplotlib.pyplot as plt

from sklearn import datasets
from sklearn import svm


alg=svm.SVC(gamma=0.001,C=100)

digits=datasets.load_digits()

xtrain,ytrain=digits.data[:1700],digits.target[:1700]

alg.fit(xtrain,ytrain)

plt.imshow(digits.images[-58],cmap=plt.cm.gray_r,interpolation="nearest")
print(alg.predict(digits.data[-58]))
plt.show()
