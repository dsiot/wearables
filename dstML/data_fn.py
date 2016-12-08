# inbuilt modules
import numpy as np
from sklearn.cross_validation import train_test_split
# defined modules
from dataexp_methods import *

def split_data(data,label,split,balance_data=True):
    (n,n_feat) = data.shape
    x_train,x_test,y_train,y_test = train_test_split(data,label,test_size = 1-split)
    if balance_data==True:
        # split x_train into both classes, 0 and 1
        n1 = np.sum(y_train)
        loc = np.where(y_train==1)[0]
        data1 = x_train[loc,:]
        label1 = y_train[loc]
        n0 = len(x_train) - n1
        loc = np.where(y_train==0)[0]
        data0 = x_train[loc,:]
        label0 = y_train[loc]
        # if data1 is minor, expand data1
        if n0>n1:
            factor = np.round(n0*1.0/n1).astype(int)
            if factor>1:
                data1 = expand_data(data1,factor,'oversample')
                label1 = np.ones((data1.shape[0]))
        # if data0 is minor, expand data0
        else:
            factor = np.round(n1*1.0/n0).astype(int)
            if factor>1:
                data0 = expand_data(data0,factor)
                label0 = np.zeros((data0.shape[0]))
        # combine data0 and data1, label0 and label1
        x_train = np.concatenate((data0,data1),axis=0)
        y_train = np.concatenate((label0,label1),axis=0)
    return x_train,x_test,y_train,y_test

def normalize(x):
	x=x-np.mean(x,axis=0)
	x=x/np.std(x,axis=0)
	return x

def prepare_data(x_train,x_test):
    x_train = x_train.astype('float32')
    x_test = x_test.astype('float32')
    x_train = normalize(x_train)
    x_test = normalize(x_test)
    return x_train,x_test


