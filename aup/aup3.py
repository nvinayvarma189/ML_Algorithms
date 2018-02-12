import theano
import numpy

#defining the independent and dependent variables
x=theano.tensor.fvector('x')
y=theano.tensor.fvector('y')

w=theano.shared(numpy.array([0.1,0.2]),'w')
#equation for algorithm
yhat=(x*w).sum()

#defining the cost function
cost=theano.tensor.pow((y-yhat),2) # (y-yhat)**2
cost=cost/2

#gradient descent algorithm
gradw=theano.tensor.grad(cost[0],w)
wn=w-0.1*gradw

f=theano.function([x,y],yhat,updates=[(w,wn)])

for i in range(30):
   output=f([1.0,1.0],[60])
   print(output)
   

