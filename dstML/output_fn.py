import numpy as np
from activations import *

def output_ann(data,weight,actv_fn):
    layer_output = []
    length = len(weight)/2
    (n_sample,in_len) = data.shape
    in_data = data
    for layer_count in range(length):
        layer_count = 2*layer_count
        out = np.zeros((n_sample,weight[layer_count].shape[1]))
        if layer_count == length:
            actv_fn = 'sigmoid'
        for i in range(n_sample):
        	out[i,:] = activation(actv_fn,np.dot(in_data[i,:][np.newaxis,:],weight[layer_count][:][:])+weight[layer_count+1][:][np.newaxis,:])
        layer_output.append(out)
        in_data = out
    return layer_output


