import numpy
import matplotlib.pyplot as plt
import theano

#our given right data which can be used for training the algo
Y=numpy.asarray([40,50,80,40,60,90,70,50,110,70])
X=numpy.asarray([10,20,30,40,50,60,70,80,90,100])
mval=0.838016098178269
cval=13.34057933552273
#defining vectors variables
x=theano.tensor.vector(name='x')
y=theano.tensor.vector(name='y')
m=theano.shared(mval,name='m')
c=theano.shared(cval,name='c')

yh=numpy.dot(x,m)+c #defining the linear regression model
k=X.shape[0]
#defining the cost function
cost=theano.tensor.sum(theano.tensor.pow(yh-y,2))/(2*k)
#calculating the gradient
djdm=theano.tensor.grad(cost,m)
djdc=theano.tensor.grad(cost,c)

#gradient descent algorithm
mn=m-0.0005*djdm
cn=c-0.0005*djdc


#defining the train function
train= theano.function([x,y],cost,updates=[(m,mn),(c,cn)])
#defining the test function part
test=theano.function([x],yh)

#now train the algo 500 times
for i in range(500):
    costval=train(X,Y)
    print(costval)
#now test the algorithm
a=numpy.linspace(1,100,20)
b=test(a)
#lets check the output
plt.plot(a,b)

for i in range(3000):
    costv2=train(X,Y)
    print(costv2)

b2=test(a)
plt.plot(a,b2)
plt.scatter(X,Y)
plt.show()
