import matplotlib.pyplot as plt
import numpy
from sklearn import linear_model

xdata=numpy.asarray([1.2,2.0,3.6,5.8,9.11,8.51,12.55,18.52,45.12,65.12])
ydata=numpy.asarray([8.9,12.56,24.21,32.12,7.56,12.65,35.65,41.1,21.4,44.88])
xdata=xdata.reshape(10,1)
ydata=ydata.reshape(10,1)
a=numpy.linspace(0,50,20)
a=a.reshape(20,1)
#definging the model
algo=linear_model.LinearRegression()
#feeding data to your model
algo.fit(xdata,ydata)
#testing and plotting the data from the model
plt.scatter(xdata,ydata,label="training data")
b=algo.predict(a)
plt.plot(a,b,label="tested response")
z=numpy.asarray([20.02])
z=z.reshape(1,1)
print(algo.predict(z))
plt.legend()
plt.show()
