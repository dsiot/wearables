import numpy as np
from dstML.data_fn import *
from dstML.output_fn import *
from scipy.special import expit
import pickle
import csv

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
    data = normalize(data[:2,:])

    # get output from SVM
    clf_svm = pickle.load(open('clf_svm.sav','rb'))
    out_svm = expit(clf_svm.decision_function(data))
    
    # get output from ANN
    clf_ann = pickle.load(open('clf_ann.sav','rb'))
    out_ann = output_ann(data,clf_ann,'relu')[-1]

    # get output from DTree
    clf_dtree = pickle.load(open('clf_dtree.sav','rb'))
    out_dtree = clf_dtree.predict_proba(data)
    out_dtree = (out_dtree[:,1]-out_dtree[:,0]+1)/2.0
    
    print 'relax = 1, stress = 0'
    print out_ann
    print out_svm,out_dtree





