# These are all the modules we'll be using later. Make sure you can import them
# before proceeding further.
from __future__ import print_function
import numpy as np
import tensorflow as tf
from six.moves import cPickle as pickle
from six.moves import range


###################################################################################################



				#Load the data descriptors



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
	x1, y1, x2, y2, x3, y3, x4, y4, x5, y5 = init_weight(960)
	
	# Training computation.
	def model(data,w1,b1,w2,b2,w3,b3,w4,b4,w5,b5):
		lay1 = tf.nn.relu(tf.matmul(data, w1) + b1)
		lay2 = tf.nn.relu(tf.matmul(lay1, w2) + b2)
		lay3 = tf.matmul(lay2, w3) + b3
		lay4 = tf.matmul(lay3, w4) + b4
		return tf.matmul(lay4, w5) + b5
	
	logits = model(tf_train_dataset,x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
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