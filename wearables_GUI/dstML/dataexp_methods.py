import numpy as np
from tqdm import tqdm
import heapq

def nearest_neighbor(feat,data,factor):
	n = len(data)
	feat = feat.astype(float)
	dist = np.zeros((n))
	for i in range(n):
		dist[i] = np.linalg.norm(feat-data[i].astype(float))
	loc = heapq.nlargest(factor,range(int(n)),dist.take)
	data_nn = []
	for i in range(factor):
		data_nn.append(data[loc[i]])
	return data_nn

# implements smote algorithm for data expansion
def smote(data,factor):
	(m,n) = data.shape
	data_exp = np.zeros((m*factor,n))
	count=0
	for i in tqdm(range(m)):
		feat = data[i,:]
		feat_nn = nearest_neighbor(feat,data,factor)
		for j in range(factor):
			alpha = np.random.uniform(0,1)
			new_feat = alpha*feat.copy() + (1-alpha)*feat_nn[j].copy()
			data_exp[count,:]=new_feat
			count=count+1
	return data_exp

# implements oversampling algorithm for data expansion
def oversample(data,factor):
	(m,n) = data.shape
	data_exp = np.zeros((m*factor,n))
	count=0
	for i in tqdm(range(m)):
		feat = data[i,:]
		for j in range(factor):
			new_feat = feat
			data_exp[count,:]=new_feat
			count=count+1
	return data_exp

def expand_data(data,factor,method):
	if method=='oversample':
		return oversample(data,factor)
	elif method=='smote':
		return smote(data,factor)

