import numpy as np
from sklearn.metrics import auc
from scipy.special import expit
from output_fn import *

def confusion_matrix(output,label,n_roc,pos_label):
    n = len(output)
    # threshold - uniform split between 0 and 1
    thresh = np.linspace(0,1,n_roc)
    fn = np.zeros((n_roc))
    fp = np.zeros((n_roc))
    # Find fn and fp for all thresh values
    for i in range(n_roc):
    	# Finding fn and fp given a threshold
    	for j in range(n):
            out_predict = 0
    	    if output[j]>=thresh[i] and thresh[i]!=1:
    	        out_predict = 1
            # if label=0 for abnormal
            if pos_label==0:
                # since abnormal is predicted as normal, its 
                # false negative, similarily for false positive
                if label[j]==0 and out_predict==1:
                    fn[i]=fn[i]+1
                elif label[j]==1 and out_predict==0:
                    fp[i]=fp[i]+1
            # if label=0 for normal
            if pos_label==1:
                if label[j]==0 and out_predict==1:
                    fp[i]=fp[i]+1
                elif label[j]==1 and out_predict==0:
                    fn[i]=fn[i]+1
    if pos_label==0:
    	n_neg = np.sum(label)
    	n_pos = n - n_neg
    else:
    	n_pos = np.sum(label)
    	n_neg = n - n_pos
    # Finding confusion matrix
    tpr = np.zeros((n_roc))
    tnr = np.zeros((n_roc))
    ppv = np.zeros((n_roc))
    npv = np.zeros((n_roc))
    acc = np.zeros((n_roc))
    for i in range(n_roc):
        true_neg = n_neg - fp[i]
        false_pos = fp[i]
        true_pos = n_pos - fn[i]
        false_neg = fn[i]
        # find ppv, npv, tpr, tnr, acc
        if true_pos==0 and false_pos==0:
            ppv[i]=1
        else:
            ppv[i] = true_pos/(true_pos+false_pos)
        if true_neg==0 and false_neg==0:
            npv[i]=1
        else:
            npv[i] = true_neg/(true_neg+false_neg)
        tpr[i] = true_pos/(true_pos+false_neg) 
        tnr[i] = true_neg/(true_neg+false_pos)
        acc[i] = (true_pos+true_neg)/n
    auc_score = auc(1-tnr,tpr)
    # store confusion matrix
    conf_matrix = {}
    conf_matrix['acc'] = acc
    conf_matrix['tpr'] = tpr
    conf_matrix['tnr'] = tnr
    conf_matrix['fpr'] = 1-tnr
    conf_matrix['ppv'] = ppv
    conf_matrix['npv'] = npv
    conf_matrix['auc'] = auc_score
    return conf_matrix

def confmatrix_dtree(clf,data,label,n_roc,pos_label):
    out = clf.predict_proba(data)
    out = (out[:,1]-out[:,0]+1)/2.0
    return confusion_matrix(out,label,n_roc,pos_label)

def confmatrix_randforest(clf,data,label,n_roc,pos_label):
    out = clf.predict_proba(data)
    out = (out[:,1]-out[:,0]+1)/2.0
    return confusion_matrix(out,label,n_roc,pos_label)

def confmatrix_svm(clf,data,label,n_roc,pos_label):
    out = clf.decision_function(data)
    out = expit(out)
    return confusion_matrix(out,label,n_roc,pos_label)

def confmatrix_ann(wt,actv_fn,data,label,n_roc,pos_label):
    out = output_ann(data,wt,actv_fn)[-1]
    out = out[:,1]-out[:,0]
    out = (out-np.min(out))/(np.max(out)-np.min(out))
    return confusion_matrix(out,label,n_roc,pos_label)

def eval_metric_stats(conf_matrix_list):    
    total_iter = len(conf_matrix_list)
    n_roc = conf_matrix_list[0]['acc'].shape[0]
    
    # evaluation metrics
    acc = np.zeros((total_iter,n_roc))
    tpr = np.zeros((total_iter,n_roc))
    tnr = np.zeros((total_iter,n_roc))
    ppv = np.zeros((total_iter,n_roc))
    npv = np.zeros((total_iter,n_roc))
    auc = np.zeros((total_iter))
    
    for i in range(total_iter):
        conf_matrix = conf_matrix_list[i]
        acc[i,:] = conf_matrix['acc']
        tpr[i,:] = conf_matrix['tpr']
        tnr[i,:] = conf_matrix['tnr']
        ppv[i,:] = conf_matrix['ppv']
        npv[i,:] = conf_matrix['npv']
        auc[i] = conf_matrix['auc']

    eval_score = {}
    loc = n_roc/2
    eval_score['acc'] = [np.mean(acc,axis=0)[loc],np.std(acc,axis=0)[loc]]
    eval_score['tpr'] = [np.mean(tpr,axis=0)[loc],np.std(tpr,axis=0)[loc]]
    eval_score['tnr'] = [np.mean(tnr,axis=0)[loc],np.std(tnr,axis=0)[loc]]
    eval_score['ppv'] = [np.mean(ppv,axis=0)[loc],np.std(ppv,axis=0)[loc]]
    eval_score['npv'] = [np.mean(npv,axis=0)[loc],np.std(npv,axis=0)[loc]]
    eval_score['auc'] = [np.mean(auc),np.std(auc)]
    
    return eval_score


