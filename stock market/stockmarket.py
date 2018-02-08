import theano
import numpy
#fvector=float type of vector
#tensor is a function of theano

#defining the independent and dependent variables
x=theano.tensor.fvector('x')#'x' implies of type x
y=theano.tensor.fvector('y')

w=theano.shared(numpy.array([0.1,0.2]),'w')
#equation for algorithm
yhat =(x*w).sum()
#defining the cost function
cost = theano.tensor.pow((y-yhat),2)
cost=cost/2

#gradient descent algorithm
gradw=theano.tensor.grad(cost[0],w)
wn=w-0.1*gradw #wnew
f=theano.function([x,y],yhat,updates=[(w,wn)])
#training the algorithm
for i in range(100):#number of times this loop exectues more will be the accuracy bcz the algorithm will be more trained
    output=f([1.0,1.0],[60])
    print(output)

    
