#regression problem using artificial neural networks using theano and numpy

import numpy
import theano
#step1 define the theano variables
x=theano.tensor.fvector(name='x')
y=theano.tensor.fvector(name='y')
wval=numpy.asarray([0.2,0.6])
w=theano.shared(wval,name='w')
#start with the neural network
yhat=theano.tensor.dot(x,w)
cost=((y[0]-yhat)**2)/2
grad=theano.tensor.grad(cost,[w])
#gradient descent
wn=w-0.1*grad[0]
#theano function
neuron=theano.function([x,y],yhat,updates=[(w,wn)])

for i in range(50):
    output=neuron([1.0,1.0],[30.0])
    print(output)
