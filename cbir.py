# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range
import random
import sys
from descriptor import *
import heapq as hq
import shutil
import os

## take query image as input
# if len(sys.argv) < 2:
# 	print('Please input the image')
# 	exit(1)
	
# query_img_file = sys.argv[1]

###################################################################################################
#Load the data descriptors
def load_data():
	with open('gist.pickle', 'rb') as f:
		save = pickle.load(f)
		gist_desc = save['gist']
		del save
		print('gist_desc: ', len(gist_desc),len(gist_desc[0]))

	with open('surf.pickle', 'rb') as f:
		save = pickle.load(f)
		surf_desc = save['surf']
		del save
		print('surf_desc: ', len(surf_desc), len(surf_desc[0][0]))

	with open('sift.pickle', 'rb') as f:
		save = pickle.load(f)
		sift_desc = save['sift']
		del save
		print('sift_desc: ', len(sift_desc), len(sift_desc[0][0]))

	with open('gabor.pickle', 'rb') as f:
		save = pickle.load(f)
		gabor_desc = save['gabor']
		del save
		print('gabor_desc: ', len(gabor_desc), len(gabor_desc[0]))

	with open('lbp.pickle', 'rb') as f:
		save = pickle.load(f)
		lbp_desc = save['lbp']
		del save
		print('lbp_desc: ', len(lbp_desc), len(lbp_desc[0]))
	
	with open('label.pickle', 'rb') as f:
		save = pickle.load(f)
		labels = save['label']
		del save
		print('labels: ', len(labels), len(labels[0]))

	with open('path.pickle', 'rb') as f:	
		save = pickle.load(f)
		paths = save['path']
		del save
		print('paths: ', len(paths))

	return gist_desc, surf_desc, sift_desc, gabor_desc, lbp_desc, labels, paths

gist_desc, surf_desc, sift_desc, gabor_desc , lbp_desc, labels, paths = load_data()

labels_index = {}
labels_sum = {}

sum = 0
labl = labels[0]
labels_index[labl] = 0
for i in range(len(labels)):
	if(labl != labels[i]):
		print(labl, sum)
		labels_sum[labl] = sum
		labl = labels[i]
		my_sum = 0
		labels_index[labl] = i
	else:
		sum += 1
labels_sum[labl] = sum

print(labels_sum)
print(labels_index)
print()


test_gist_desc = []
test_surf_desc = []
test_sift_desc = []
test_gabor_desc = []
test_lbp_desc = []
test_labels = []

for k,v in labels_index.items():
	print(k,v)
	start = v
	end = v + max(1,labels_sum[k]//10)
	for i in gist_desc[start:end]:
		test_gist_desc.append(i)
	for i in surf_desc[start:end]:
		test_surf_desc.append(i)
	for i in sift_desc[start:end]:
		test_sift_desc.append(i)
	for i in gabor_desc[start:end]:
		test_gabor_desc.append(i)
	for i in lbp_desc[start:end]:
		test_lbp_desc.append(i)
	for i in labels[start:end]:
		test_labels.append(i)
	del gist_desc[start:end]
	del surf_desc[start:end]
	del sift_desc[start:end]
	del gabor_desc[start:end]
	del lbp_desc[start:end]
	del labels[start:end]
	
	sum = 0
	labels_index = {}
	labels_sum = {}
	labl = labels[0]
	labels_index[labl] = 0
	for i in range(len(labels)):
		if(labl != labels[i]):
			labels_sum[labl] = sum
			labl = labels[i]
			sum = 0
			labels_index[labl] = i
		else:
			sum += 1
	labels_sum[labl] = sum

print(labels_sum)
print(labels_index)
print("*************************Data Loaded**********************************")

###################################################################################################
#Graph:
num_labels = 10
batch_size = 1
num_hidden_nodes = 100
num_hidden_nodes2 = 50

#init weight for different network
def init_weight(input_size):
	w1 = tf.Variable(tf.truncated_normal([input_size, num_hidden_nodes],stddev=0.1))
	b1 = tf.Variable(tf.zeros([num_hidden_nodes]))
	
	w2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes2],stddev=0.1))
	b2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
		
	w3 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes2],stddev=0.1))
	b3 = tf.Variable(tf.zeros([num_hidden_nodes2]))

	w4 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes2],stddev=0.1))
	b4 = tf.Variable(tf.zeros([num_hidden_nodes2]))

	w5 = tf.Variable(tf.truncated_normal([num_hidden_nodes2, num_hidden_nodes2],stddev=0.1))
	b5 = tf.Variable(tf.zeros([num_hidden_nodes2]))
	#return w1,b1,w2,b2,w3,b3,w4,b4,w5,b5
	return (w1,w2,w3,w4,w5),(b1,b2,b3,b4,b5)



graph = tf.Graph()
with graph.as_default():
	tf_train_gist = tf.placeholder(tf.float32,shape=(batch_size, 2*960))
	tf_train_sift = tf.placeholder(tf.float32,shape=(batch_size, 2*200))
	tf_train_surf = tf.placeholder(tf.float32,shape=(batch_size, 2*200))
	#tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size))
	tf_train_gabor = tf.placeholder(tf.float32,shape=(batch_size, 2*256))
	tf_train_lbp = tf.placeholder(tf.float32,shape=(batch_size, 2*256))

	tf_test_gist = tf.constant(np.array(test_gist_desc))
	tf_test_surf = tf.constant(np.array(test_surf_desc))
	tf_test_sift = tf.constant(np.array(test_sift_desc))
	tf_test_gabor = tf.constant(np.array(test_gabor_desc))
	tf_test_lbp = tf.constant(np.array(test_lbp_desc))
	global_step = tf.Variable(0)

	# Init Variables.
	gist_w, gist_b = init_weight(2*960)
	surf_w, surf_b = init_weight(2*200)
	sift_w, sift_b = init_weight(2*200)
	gabor_w, gabor_b = init_weight(2*256)
	lbp_w, lbp_b = init_weight(2*256)

	sum_w = tf.Variable(tf.truncated_normal([5*num_hidden_nodes2, 1],stddev=0.1))
	sum_b = tf.Variable(tf.zeros([1]))
	# Training computation.
	def model(data,W,B):
		lay1 = tf.nn.relu(tf.matmul(data, W[0]) + B[0])
		lay2 = tf.nn.relu(tf.matmul(lay1, W[1]) + B[1])
		lay3 = tf.nn.relu(tf.matmul(lay2, W[2]) + B[2])
		lay4 = tf.nn.relu(tf.matmul(lay3, W[3]) + B[3])
		return tf.nn.relu(tf.matmul(lay4, W[4]) + B[4])
	
	gist_logits = model(tf_train_gist, gist_w, gist_b)
	surf_logits = model(tf_train_sift, surf_w, surf_b)
	sift_logits = model(tf_train_sift, sift_w, sift_b)
	gabor_logits = model(tf_train_gabor, gabor_w, gabor_b)
	lbp_logits = model(tf_train_lbp, lbp_w, lbp_b)

	concat = tf.stack([gist_logits, surf_logits, sift_logits, gabor_logits, lbp_logits], axis=1)
	concat = tf.reshape(concat, [batch_size,150]) 
	similarity = tf.matmul(concat, sum_w) + sum_b
	similarity = tf.reshape(similarity,(1,-1))[0]
	
	#Calculate the loss
	L = []
	for i in range(batch_size):#similarity[i+1] ==> x,x-
		a = similarity[i]
		i += 1
		if(i < batch_size):
			b = similarity[i]
			s = 1 + b - a
			r = 0
			r = tf.cond(s>0,lambda: s,lambda: tf.add(0.,0.))
			L.append(r)
			#L += max(0, s)

	# loss = tf.reduce_mean(L)
	#loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
	# Optimizer.
	# optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)

########################################################################################################


def generate_batch(query_img_features, cur_img_ind):
	train_gist = []
	train_sift = []
	train_surf = []
	train_gabor = []
	train_lbp = []

	train_gist.append(np.concatenate((gist_desc[cur_img_ind] , query_img_features['gist']),axis=0))
	train_sift.append(np.concatenate((sift_desc[cur_img_ind][0], query_img_features['sift'][0]),axis=0))
	train_surf.append(np.concatenate((surf_desc[cur_img_ind][0], query_img_features['surf'][0]),axis=0))
	train_gabor.append(np.concatenate((surf_desc[cur_img_ind][0], query_img_features['gabor'][0]),axis=0))
	train_lbp.append(np.concatenate((surf_desc[cur_img_ind][0], query_img_features['lbp'][0]),axis=0))

	return np.array(train_gist), np.array(train_sift), np.array(train_surf), np.array(train_gabor), np.array(train_lbp)


def gen_query_features(img, bowDiction_sift, bowDiction_surf):
	f = {}
	f['sift'] = bow_feature_extract_sift(bowDiction_sift, img)
	f['surf'] = bow_feature_extract_surf(bowDiction_surf, img)
	f['gist'] = gist_descriptor(img)
	f['gabor'] = gabor(img)
	f['lbp'] = local_binary_pattern(img)
	return f

bowDiction_sift = load_sift_bow_diction('Sift_Voc')
print('Done Sift dict')
# bowDiction_surf = bag_of_words_surf(pics_only, v)
bowDiction_surf = load_surf_bow_diction('Surf_Voc')
print('Done Surf dict')


def load_test_pics(path='img/'):
	res = []
	with open('men_list_test.txt') as f:
		for line in f:
			line = line.strip()
			s = line.split('/')
			
			img = misc.imread(path + line)
			
			res += [(img, s[3], s[1] + '/' + s[2], s[4], line)]
	return res


del sum
eval_res = []
top_n = 5

### for visualization
dst_root = 'retrieval'
shutil.rmtree(dst_root, True)
os.makedirs(dst_root)
src_root = 'img/'

with tf.Session(graph=graph) as sess:
	saver = tf.train.Saver()
	saver.restore(sess, "./model.ckpt")
	print("Model Loaded")

	test_num = -1
	for test_tuple in load_test_pics():
		test_num += 1
		retrievals = []
		query_img_features = gen_query_features(test_tuple[0], bowDiction_sift, bowDiction_surf)
	
		for i in range(len(gist_desc)):
			a,b,c,d,e = generate_batch(query_img_features, i)
			# print("***************************",a.shape,b.shape,c.shape)
			# Prepare a dictionary telling the session where to feed the minibatch.
			feed_dict = {tf_train_gist: a, tf_train_sift: b,tf_train_surf: c, tf_train_gabor: d, tf_train_lbp:e}
			sim = sess.run([similarity], feed_dict=feed_dict)
			# print(i, 'Sim =', sim)

			hq.heappush(retrievals, (sim[0][0], i))
			if len(retrievals) > top_n:
				hq.heappop(retrievals)
		
		correct = sum([1 for j in [labels[u[1]] for u in retrievals] if j == test_tuple[2]])
		eval_res += [correct / top_n]

		print('[query label]', test_tuple[2])
		print('[sims]', retrievals)
		print('[labels]', correct, [labels[i[1]] for i in retrievals])
		print()

		
		dst_cur = dst_root + '/test_' + str(test_num) + '_c' + str(correct)
		os.makedirs(dst_cur)

		src = src_root + '/' + test_tuple[4]
		dst = dst_cur + '/query_' + test_tuple[2].replace('/', '-') + '.jpg'
		shutil.copyfile(src, dst)
		
		for j in range(len(retrievals)):
			src = src_root + '/' + paths[retrievals[j][1]]
			dst = dst_cur + '/' + str(j) + '_' + labels[retrievals[j][1]].replace('/', '-') + '.jpg'
			shutil.copyfile(src, dst)
		

print('[Mean Accuracy]', sum(eval_res) / len(eval_res))
