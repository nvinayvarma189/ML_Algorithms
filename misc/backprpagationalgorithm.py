import numpy
#input data
x=numpy.asarray([[0,0,1],
                 [0,1,1],
                 [1,0,1],
                 [1,1,1]])
#output data
yd=numpy.asarray([[0],[1],[1],[0]])
#we are seeding the random values so that everytime we can get
#uniformly distributed weights
numpy.random.seed(1)

#define the weights
#w0= weight between the input layer and the hidden layer
w0=2*numpy.random.random((3,4))-1
#w1=weight between the hidden ;ayer and the output layer
w1=2*numpy.random.random((4,1))-1
for i in range(30000):
    #feed forward the input data from inputs to hidden and hidden to outputs
    l0=x;
    l1=1/(1+numpy.exp(-(numpy.dot(l0,w0))))
    l2=1/(1+numpy.exp(-(numpy.dot(l1,w1))))

    #error calculation on output neuron
    outputerr=yd-l2

    #delat value on the output neuron
    delta2=outputerr*(l2*(1-l2))
    #error on layer 1
    l1err=delta2.dot(w1.T)
    delta1=l1err*(l1*(1-l1))

    #apply the gradient descent formula over here
    w1=w1+1*l1.T.dot(delta2)
    w0=w0+1*l0.T.dot(delta1)

    if (i%500)==0:
        print("Error= "+str(numpy.mean(numpy.abs(outputerr))))
              
print('the output after 10000 iterations is')
print(l2)







    





    





    
          
