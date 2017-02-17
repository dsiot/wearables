import numpy as np

def sigmoid(X):
    return 1/(1+np.exp(-X))

def relu(X):
	X=X
	x=X.shape[1]
	y=np.zeros((1,x))
	for i in range(x):
		if X[0,i]>=0:
			y[0,i]=X[0,i]
	return y

def activation(x,y):
	if x=='sigmoid':
		return sigmoid(y)
	elif x=='linear':
		return linear(y)
	elif x=='relu':
		return relu(y)

