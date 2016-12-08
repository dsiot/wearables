# inbuilt modules
import numpy as np
from scipy.io import loadmat,savemat
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils
from keras.regularizers import l1l2,activity_l1l2
from tqdm import tqdm
import csv
# defined modules
from dstML.eval_metrics import *
from dstML.data_fn import *

def get_data(file_path):
    # read data from csv file
    csv_data = []
    with open('data.csv','rb') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)
    
    # extract data and labels
    n = len(csv_data)
    n_feat = len(csv_data[0])-2
    data = np.zeros((n,n_feat))
    label = np.zeros((n))
    # if reading=stress, label=0 ; reading=relax, label=1
    for i in range(n):
        data[i,:] = csv_data[i][1:7]
        label[i] = 1 if csv_data[i][-1]=='relax' else 0
    return data,label

if __name__=='__main__':
	
    # load data
    file_path = 'data.csv'
    data,label = get_data(file_path) 

    # model parameters 
    nb_classes = 2
    lr = 1e-3
    total_iter = 1
    n_roc = 55
    split = 0.8
    input_shape = data.shape[1]
    nb_epoch = 100
    actv_fn = 'relu'
    batch_size = 1
    adam = Adam(lr=lr)
    adam = SGD(lr=lr,momentum=0.9,decay=1e-6)
    conf_matrix_list_train = []
    conf_matrix_list_test = []
    
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):
        
        # get training, testing data with data balancingi
        x_train,x_test,y_train,y_test = split_data(data,label,split,balance_data=True)
        # prepare the data for efficient modelling
        x_train,x_test = prepare_data(x_train,x_test)
        Y_train = np_utils.to_categorical(y_train, nb_classes)
        Y_test = np_utils.to_categorical(y_test, nb_classes)
        # model the network
        # network : 6-3-2
        model = Sequential()
        model.add(Dense(3,input_dim=input_shape))
        model.add(Activation(actv_fn))
        model.add(Dense(nb_classes))
        model.add(Activation('sigmoid'))
        # train the network
        model.compile(loss='binary_crossentropy',optimizer=adam,metrics=["accuracy"])
        out,wt = model.fit(x_train,Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose=0 ,validation_data=(x_test, Y_test))
        best_wt = wt[-1]
        # get confusion matrix
        conf_matrix_list_train.append(confmatrix_ann(best_wt,actv_fn,x_train,y_train,n_roc,pos_label=0))
        conf_matrix_list_test.append(confmatrix_ann(best_wt,actv_fn,x_test,y_test,n_roc,pos_label=0))
    
    # get evaluation scores mean and std
    eval_score_train = eval_metric_stats(conf_matrix_list_train)
    eval_score_test = eval_metric_stats(conf_matrix_list_test)   
    print 'training scores : '
    print 'acc = ',round(eval_score_train['acc'][0],3),' ',round(eval_score_train['acc'][1],3)
    print 'tpr = ',round(eval_score_train['tpr'][0],3),' ',round(eval_score_train['tpr'][1],3)
    print 'tnr = ',round(eval_score_train['tnr'][0],3),' ',round(eval_score_train['tnr'][1],3)
    print 'auc = ',round(eval_score_train['auc'][0],3),' ',round(eval_score_train['auc'][1],3)
    print 'testing scores : '
    print 'acc = ',round(eval_score_test['acc'][0],3),' ',round(eval_score_test['acc'][1],3)
    print 'tpr = ',round(eval_score_test['tpr'][0],3),' ',round(eval_score_test['tpr'][1],3)
    print 'tnr = ',round(eval_score_test['tnr'][0],3),' ',round(eval_score_test['tnr'][1],3)
    print 'auc = ',round(eval_score_test['auc'][0],3),' ',round(eval_score_test['auc'][1],3)
 
