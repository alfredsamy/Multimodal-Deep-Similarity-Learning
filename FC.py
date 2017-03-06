# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


###################################################################################################
#Load the data descriptors
def load_data():
	with open('gist', 'rb') as f:
		save = pickle.load(f)
		gist_desc = save['gist']
		del save
		print('gist_desc: ', gist_desc.shape)

	with open('surf', 'rb') as f:
		save = pickle.load(f)
		surf_desc = save['surf']
		del save
		print('surf_desc: ', surf_desc.shape)

	with open('sift', 'rb') as f:
		save = pickle.load(f)
		surf_desc = save['sift']
		del save
		print('sift_desc: ', sift_desc.shape)

	with open('labels', 'rb') as f:
		save = pickle.load(f)
		labels = save['labels']
		del save
		print('labels: ', labels.shape)

	return gist_desc, surf_desc, sift_desc, labels

gist_desc, surf_desc, sift_desc, labels = load_data()
###################################################################################################
#					TO BE Removed just for reference
#Reformat into a shape that's more adapted to the models we're going to train:
#	data as a flat matrix,
#	labels as float 1-hot encodings.

# def reformat(dataset, labels):
# 	dataset = dataset.reshape((-1, image_size * image_size)).astype(np.float32)
# 	# Map 0 to [1.0, 0.0, 0.0 ...], 1 to [0.0, 1.0, 0.0 ...]
# 	labels = (np.arange(num_labels) == labels[:,None]).astype(np.float32)
# 	return dataset, labels


# train_dataset, train_labels = reformat(train_dataset, train_labels)
# valid_dataset, valid_labels = reformat(valid_dataset, valid_labels)
# test_dataset, test_labels = reformat(test_dataset, test_labels)
# print('Training set', train_dataset.shape, train_labels.shape)
# print('Validation set', valid_dataset.shape, valid_labels.shape)
# print('Test set', test_dataset.shape, test_labels.shape)

#Graph:
num_labels = 10
batch_size = 16
num_hidden_nodes = 100
num_hidden_nodes2 = 50

#init weight for different network
def init_weight(input_size):
	w1 = tf.Variable(tf.truncated_normal([input_size, num_hidden_nodes],stddev=0.1))
	b1 = tf.Variable(tf.zeros([num_hidden_nodes]))
	
	w2 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes2],stddev=0.1))
	b2 = tf.Variable(tf.zeros([num_hidden_nodes2]))
		
	w3 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes2],stddev=0.1))
	b3 = tf.Variable(tf.zeros([num_hidden_nodes2]))

	w4 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_hidden_nodes2],stddev=0.1))
	b4 = tf.Variable(tf.zeros([num_hidden_nodes2]))

	w5 = tf.Variable(tf.truncated_normal([num_hidden_nodes, num_labels],stddev=0.1))
	b5 = tf.Variable(tf.zeros([num_labels]))
	return w1,b1,w2,b2,w3,b3,w4,b4,w5,b5


graph = tf.Graph()
with graph.as_default():
	tf_train_dataset = tf.placeholder(tf.float32,shape=(batch_size, 1000))#TO BE Chanded
	tf_train_labels = tf.placeholder(tf.float32, shape=(batch_size, num_labels))
	tf_valid_dataset = tf.constant(valid_dataset)
	tf_test_dataset = tf.constant(test_dataset)
	global_step = tf.Variable(0)

	# Init Variables.
	gist_w = []
	gist_b = []
	gist_w[1], gist_b[1], gist_w[2], gist_b[2], gist_w[3], gist_b[3], gist_w[4], gist_b[4], gist_w[5], gist_b[5] = init_weight(960)
	
	surf_w = []
	surf_b = []
	surf_w[1], surf_b[1], surf_w[2], surf_b[2], surf_w[3], surf_b[3], surf_w[4], surf_b[4], surf_w[5], surf_b[5] = init_weight(1000)

	sift_w = []
	sift_b = []
	sift_w[1], sift_b[1], sift_w[2], sift_b[2], sift_w[3], sift_b[3], sift_w[4], sift_b[4], sift_w[5], sift_b[5] = init_weight(1000)
	
	# Training computation.
	def model(data,W,B):
		lay1 = tf.nn.relu(tf.matmul(data, W[1]) + B[1])
		lay2 = tf.nn.relu(tf.matmul(lay1, W[2]) + B[2])
		lay3 = tf.nn.relu(tf.matmul(lay2, W[3]) + B[3])
		lay4 = tf.nn.relu(tf.matmul(lay3, W[4]) + B[4])
		return tf.nn.relu(tf.matmul(lay4, W[5]) + B[5])
	
	gist_logits = model(tf_train_dataset,gist_w,gist_b)
	surf_logits = model(tf_train_dataset,surf_w,surf_b)
	sift_logits = model(tf_train_dataset,sift_w,sift_b)

	sum_w = tf.Variable(tf.truncated_normal([3*num_labels, 1],stddev=0.1))
	sum_b = tf.Variable(tf.zeros([1]))
	
	concat = np.concatenate((gist_logits, surf_logits, sift_logits), axis=0)
	similarity = tf.matmul(concat, sum_w) + sum_b

	# L t ((x t , x +t , x t ); S) = max(0, S(x t , x t ) − S(x t , x t ) + 1),
	L = 0
	for i in range(len(similarity)):
		L += 1 + similarity[i+1] - similarity[i]
		i++

	loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits, tf_train_labels))
	
	# Optimizer.
	optimizer = tf.train.GradientDescentOptimizer(0.1).minimize(loss)
	# learning_rate = tf.train.exponential_decay(0.5, global_step, 1000, 0.65, staircase=True)
	# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss, global_step=global_step)
	
	# Predictions for the training, validation, and test data.
	train_prediction = tf.nn.softmax(logits)
	valid_prediction = tf.nn.softmax(model(tf_valid_dataset))
	test_prediction = tf.nn.softmax(model(tf_test_dataset))


#Session

num_steps = 5001
def accuracy(predictions, labels):
	return (100.0 * np.sum(np.argmax(predictions, 1) == np.argmax(labels, 1))
			 / predictions.shape[0])

with tf.Session(graph=graph) as session:
	tf.initialize_all_variables().run()
	print("Initialized")
	for step in range(num_steps):
		offset = random.randint(0,train_labels.shape[0] - batch_size)
		# Generate a minibatch.
		batch_data = train_dataset[offset:(offset + batch_size), :]
		batch_labels = train_labels[offset:(offset + batch_size), :]
		
		# Prepare a dictionary telling the session where to feed the minibatch.
		feed_dict = {tf_train_dataset : batch_data, tf_train_labels : batch_labels}
		_, l, predictions = session.run([optimizer, loss, train_prediction], feed_dict=feed_dict)
		if (step % 500 == 0):
			print("Minibatch loss at step %d: %f" % (step, l))
			print("Minibatch accuracy: %.1f%%" % accuracy(predictions, batch_labels))
			print("Validation accuracy: %.1f%%" % accuracy(valid_prediction.eval(), valid_labels))
			print()
	print("Test accuracy: %.1f%%" % accuracy(test_prediction.eval(), test_labels))