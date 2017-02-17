# Required Modules
import numpy as np
import csv
from dstML.data_fn import *
from dstML.output_fn import *
from scipy.special import expit
import pickle
from sklearn import svm,tree

from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD, Adam
from keras.utils import np_utils

import tkMessageBox
import datetime,os,time

# Display error messages
def error_handler(err_name):
    # Define errors
    error = {}
    error['No Train data'] = ['Error: No Train Data file','Please select Train data']
    error['No Test data'] = ['Error: No validation Data file','Please select validation data']
    error['Incompatible data'] = ['Error: Data Incompatible','Data file not compatible. File must have atleast 7 columns']
    tkMessageBox.showinfo(error[err_name][0],error[err_name][1])

# Display instructions in Part 1
def displ_instructions():
    text = 'I/O FORMAT DETAILS \n\n'+\
            'INPUT FORMAT :\n'+\
            '1. The input file must be a .csv file.\n'+\
            '2. Each row must contain a timestamp followed by feature vector of length 6 in the first 7 columns. The remaining columns can be filled with any information.\n\n'+\
            'OUTPUT FORMAT :\n'+\
            '1. The output file will be generated in the directory of the validation data.\n'+\
            '2. Format of output file name : \'validation_filename\'_\'date\'_\'time\'.csv.\n'+\
            '3. The output file contains the input file data, along with the prediction label (\'stress\' or \'relax\') appended after the last column.\n' 
    tkMessageBox.showinfo('Instructions',text)

# Extract data from imput csv file
def get_data(file_path):
    # read data from csv file
    csv_data = []
    with open(file_path,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)
    # check compatibility of csv file
    if len(csv_data[0])<7:
        error_handler('Incompatible data')
        return
    # extract data and labels
    n = len(csv_data)
    n_feat = 6
    data = np.zeros((n,n_feat))
    label = np.zeros((n))
    # if reading=stress, label=0 ; reading=relax, label=1
    for i in range(n):
        data[i,:] = csv_data[i][1:7]
        label[i] = 1 if csv_data[i][-1]=='relax' else 0
    return data,label

# Save output in a csv file
def save_output(path,out):
    # read data from csv file
    csv_data = []
    with open(path,'rb') as f:
        reader = csv.reader(f)
        for row in reader:
            csv_data.append(row)
    n_data = len(csv_data)
    n = len(out)
    # add output column to the data
    for i in range(n):
        if out[i]==0:
            out_label = 'stress'
        else:
            out_label = 'relax'
        csv_data[i].append(out_label)
    # create write file using date and time in test directory
    write_dir = path[:-4]+'_'
    out_name = str(datetime.datetime.now())
    out_name = out_name.replace(' ','_')
    out_name = out_name[:19]+'.csv'
    write_path = write_dir+out_name
    # create output file
    with open(write_path,'wb') as f:
        writer = csv.writer(f)
        for i in range(n):
            writer.writerow(csv_data[i])
    f.close()
    return write_path

# Train the desired model with the given training data
def train_model(data,label,model):

    if model=='ann':
	# model parameters 
	nb_classes = 2
	lr = 1e-3
	split = 0.8
	input_shape = data.shape[1]
	nb_epoch = 100
	actv_fn = 'relu'
	batch_size = 1
	adam = Adam(lr=lr)
	#adam = SGD(lr=lr,momentum=0.9,decay=1e-6)
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
	model.set_weights(pickle.load(open('models/clf_ann.sav','rb')))
	# train the network
	model.compile(loss='binary_crossentropy',optimizer=adam,metrics=["accuracy"])
	out,wt = model.fit(x_train,Y_train,batch_size=batch_size, nb_epoch=nb_epoch, verbose=0 ,validation_data=(x_test, Y_test))
	best_wt = wt[-1]
	return best_wt

    elif model=='svm':
	# model parameters 
	nb_classes = 2
	lr = 1e-3
	split = 0.8
	# number of times data is randomized
        # get training, testing data with data balancing
        x_train,x_test,y_train,y_test = split_data(data,label,split,balance_data=True)
        # prepare the data for efficient modelling
        x_train,x_test = prepare_data(x_train,x_test)
        # train the network !
        clf = svm.SVC(gamma=0.01,C=10,kernel='rbf',verbose=False)
        clf = clf.fit(x_train,y_train)
        return clf
    
    elif model=='dtree':
	# model parameters 
	nb_classes = 2
	lr = 1e-3
	split = 0.8
	# number of times data is randomized
        # get training, testing data with data balancing
        x_train,x_test,y_train,y_test = split_data(data,label,split,balance_data=True)
        # prepare the data for efficient modelling
        x_train,x_test = prepare_data(x_train,x_test)
        # train the network !
        clf = tree.DecisionTreeClassifier(criterion='gini',max_depth=4,min_samples_split=4,min_samples_leaf=5,max_features=0.7)
        clf = clf.fit(x_train,y_train)
	return clf

def get_output(train_flag,model_flag,train_path,test_path,status):
    
    # check data
    if test_path=='':
        error_handler('No Test data')
        return
    if(train_flag==1 and train_path==''):
        error_handler('No Train data')
        return
    # load test data
    test_data,test_label = get_data(test_path)
    test_data = normalize(test_data[:,:])
    # Train the model and get output
    if train_flag==1:
	# load train data
    	train_data,train_label = get_data(train_path)
	if model_flag==0:
	    clf_ann = train_model(train_data,train_label,'ann')
	    out = output_ann(test_data,clf_ann,'relu')[-1]
	    out = np.round((out[:,1]-out[:,0]+1)/2.0)
	elif model_flag==1:
	    clf_svm = train_model(train_data,train_label,'svm')
	    out = expit(clf_svm.decision_function(test_data))
            out = np.round(out)
	elif model_flag==2:
	    clf_dtree = train_model(train_data,train_label,'dtree')
	    out_dtree = clf_dtree.predict_proba(test_data)
	    out = np.round((out_dtree[:,1]-out_dtree[:,0]+1)/2.0)
    # get the output using pretrained models
    else:
        if model_flag==0:
            # get output from ANN
            clf_ann = pickle.load(open('models/clf_ann.sav','rb'))
            out = output_ann(test_data,clf_ann,'relu')[-1]
            out = np.round((out[:,1]-out[:,0]+1)/2.0)
            
        elif model_flag==1: 
            clf_svm = pickle.load(open('models/clf_svm.sav','rb'))
            out = expit(clf_svm.decision_function(test_data))
            out = np.round(out)
        elif model_flag==2:
            clf_dtree = pickle.load(open('models/clf_dtree.sav','rb'))
            out_dtree = clf_dtree.predict_proba(test_data)
            out = np.round((out_dtree[:,1]-out_dtree[:,0]+1)/2.0)
    write_path = save_output(test_path,out)
    text =  'The task is complete ! \nPlease check the folder of the validation data for the output file. \nThe path of the generated output file is :\n\n '+write_path
    tkMessageBox.showinfo('Done !',text)
    
# Text of the GUI
part1_title = 'Wearables Project GUI'
part1_description = '\nThis GUI is the implementation of the Wearables Project Stress Detection Algorithm. \nPlease find the source code of the algorithm at : https://github.com/dsiot/wearables\n Please click the \'Instructions\' button for details on I/O of the GUI.\n'

part2_title = 'Model Type'
part2_description = 'Select whether training is to be performed or not. \nIn the latter case, pre-trained models\n are used for the classification task'

part3_title = 'Select Classifier'
part3_description = 'Select the desired classifier type'

part4_title = 'Data Selection'
part4_description = 'Select the data to be used for training and validation.\n Training data need not be selected if\n \'Pretrained\' is chosen in the \'Model type\' section.\n'

part5_title = 'Perform Stress Detection'
part5_description = 'Click \'Classify !\' to perform the stress detection.\n'

















