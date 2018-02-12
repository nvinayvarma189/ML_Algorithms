import numpy
import matplotlib.pyplot as plt
from sklearn import linear_model

xdata=numpy.array([75,45,78,85,45,52,12,63,96,42,21,23,24,14,10])
ydata=numpy.array([2,4,5,6,8,9,12,14,23,12,21,25,24,26,19])

xdata=xdata.astype('float32')
ydata=ydata.astype('float32')
xdata=xdata.reshape(15,1)
ydata=ydata.reshape(15,1)

alg=linear_model.LinearRegression()
alg.fit(xdata,ydata)

t=numpy.array([10,20,30,40,50,60,70,80,90,95,100,110,115,120,125]).astype('float64')
t=t.reshape(15,1)
b=alg.predict(t)
plt.scatter(xdata,ydata)
plt.plot(t,b)
plt.show()
