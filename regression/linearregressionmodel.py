import numpy
import theano
import matplotlib.pyplot as plt

xdata=numpy.asarray([1.2,2.0,3.6,5.8,9.11,8.51,12.55,18.52,45.12,65.12])
ydata=numpy.asarray([8.9,12.56,24.21,32.12,7.56,12.65,35.65,41.1,21.4,44.88])

#step1 is defining the LR model
x=theano.tensor.vector(name='x')
y=theano.tensor.vector(name='y')
m=theano.shared(numpy.random.randn(),name='m')
c=theano.shared(numpy.random.randn(),name='c')
yh=numpy.dot(x,m)+c

#step2 is calculating the cost function and defining gradient decent
n=xdata.size
cost=theano.tensor.sum((y-yh)**2)/(2*n)
gradm=theano.tensor.grad(cost,m)
gradc=theano.tensor.grad(cost,c)
mn=m-0.0005*gradm
cn=c-0.0005*gradc

#step3 is to define the train and test funcitons
train=theano.function([x,y],cost,updates=[(m,mn),(c,cn)])
test=theano.function([x],yh)

#step4 is to train the algorithm i times and check testing
for i in range(1000):
    costm=train(xdata,ydata)
    print(costm)
    
a=numpy.linspace(0,70,20)
b=test(a)

print('final value of m is '+str(m.get_value()))
print('final value of c is '+str(c.get_value()))
print(test([20.02]))
plt.scatter(xdata,ydata)
plt.plot(a,b)
plt.show()
