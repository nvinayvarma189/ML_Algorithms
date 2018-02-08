import theano
import numpy
import matplotlib.pyplot as plt

xdata=numpy.array([75,45,78,85,45,52,12,63,96,42,21,23,24,14,10])
ydata=numpy.array([2,4,5,6,8,9,12,14,23,12,21,25,24,26,19])

x=theano.tensor.fvector('x')
y=theano.tensor.fvector('y')

w1=theano.shared(0.1,'w1')
w0=theano.shared(0.2,'w2')
#algorithm can be defined as--
yhat=w0*x+w1

k=xdata.size
#cost function
cost=theano.tensor.sum(theano.tensor.pow((y-yhat),2)/(2*k))

#gradient descent algorithm
gradw0=theano.tensor.grad(cost,w0)
gradw1=theano.tensor.grad(cost,w1)
w0n=w0-0.001*gradw0
w1n=w1-0.001*gradw1

train=theano.function([x,y],cost,updates=[(w0,w0n),(w1,w1n)])
for i in range(100):
    costval=train(xdata,ydata)
    print(costval)

plt.scatter(xdata,ydata)
plt.show()
