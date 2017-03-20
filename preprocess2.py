# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
#import matplotlib.pyplot as plt
import numpy as np
import os
import sys
import tarfile
#from IPython.display import display, Image
from scipy import ndimage
from sklearn.linear_model import LogisticRegression
from six.moves.urllib.request import urlretrieve
from six.moves import cPickle as pickle


def unpickle(file):
	print("unpicle" + file)
	import cPickle
	fo = open(file, 'rb')
	dict = cPickle.load(fo)
	fo.close()
	return dict

def rgb(data):
	# g = np.zeros((len(data),1024))
	# for i in range(len(data)):
	# 	l = len(data[i])/3
	# 	for j in range(l):
	# 		g[i][j] =( data[i][j] + data[i][l+j] + data[i][2*l+j] )/3.0
	# return g
	data = data.reshape((len(data),3, 32, 32))
	g = np.zeros((len(data),32,32,3), dtype=int)	
	for i in range(len(data)):
		for j in range(32):
			for k in range(32):
				g[i][j][k]= (data[i][0][j][k] ,data[i][1][j][k], data[i][2][j][k])
	return g


d = unpickle("Batch/data_batch_1")
data = d['data']
label = d['labels']

for i in range(2,6):
	d0 = unpickle("Batch/data_batch_"+str(i))
	data = np.append(data,d0['data'], axis=0)
	label = label + d0['labels']

print("data lenght: " + str(len(data)))
print("label lenght: " + str(len(label)))


train_dataset = np.array(data, dtype=int)
train_labels = label


d = unpickle("Batch/test_batch")
test_dataset = a = np.array(d['data'], dtype=int)
test_labels = d['labels']
print("test lenght: " + str(len(test_dataset)))
print("label lenght: " + str(len(test_labels)))

train_dataset = rgb(train_dataset)
test_dataset = rgb(test_dataset)

f = open("pickle_file_data2", 'wb')
save = {
    'train_dataset': train_dataset,
    'train_labels': train_labels,
    'test_dataset': test_dataset,
    'test_labels': test_labels,
}

pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
f.close()