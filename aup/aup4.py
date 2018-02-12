import theano
import numpy
import matplotlib.pyplot as plt

xdata=numpy.array([75,45,78,85,45,52,12,63,96,42,21,23,24,14,10])
ydata=numpy.array([2,4,5,6,8,9,12,14,23,12,21,25,24,26,19])

xdata=xdata.astype('float32')
ydata=ydata.astype('float32')

x=theano.tensor.fvector('x')
y=theano.tensor.fvector('y')

w1=theano.shared(0.1,'w1')
w0=theano.shared(0.2,'w0')
#algorithm can be defined as -
yhat=w0*x+w1

k=xdata.size
#cost function
cost=theano.tensor.sum(theano.tensor.pow((y-yhat),2))/(2*k)

#gradient descent algorithm
gradw0=theano.tensor.grad(cost,w0)
gradw1=theano.tensor.grad(cost,w1)
w1n=w1-0.0006*gradw1
w0n=w0-0.0006*gradw0

train=theano.function([x,y],cost,updates=[(w0,w0n),(w1,w1n)])

test=theano.function([x],yhat)

for i in range(10000):
   costval=train(xdata,ydata)
   print(costval)

t=numpy.array([1,10,20,30,40,50,60,70,80,90]).astype('float32')
output=test(t)

plt.scatter(xdata,ydata)
plt.plot(t,output)
plt.show()



