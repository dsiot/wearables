# inbuilt modules
import numpy as np
from scipy.io import loadmat,savemat
import matplotlib.pyplot as plt
from tqdm import tqdm
from sklearn import svm
import csv
# manual modules
from dstML.eval_metrics import *
from dstML.data_fn import *
import pickle

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
    conf_matrix_list_train = []
    conf_matrix_list_test = []
    # number of times data is randomized
    for n_iter in tqdm(range(total_iter)):
        
        # get training, testing data with data balancing
        x_train,x_test,y_train,y_test = split_data(data,label,split,balance_data=True)
        # prepare the data for efficient modelling
        x_train,x_test = prepare_data(x_train,x_test)
        # train the network !
        clf = svm.SVC(gamma=0.01,C=100,kernel='rbf',verbose=False)
        clf = clf.fit(x_train,y_train)
        conf_matrix_list_train.append(confmatrix_svm(clf,x_train,y_train,n_roc,pos_label=0))
        conf_matrix_list_test.append(confmatrix_svm(clf,x_test,y_test,n_roc,pos_label=0))

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

       

