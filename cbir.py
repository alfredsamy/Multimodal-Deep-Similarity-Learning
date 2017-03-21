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


if len(sys.argv) < 2:
	print('Please input the image')
	exit(1)
	
query_img_file = sys.argv[1]

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

	return gist_desc, surf_desc, sift_desc, gabor_desc, lbp_desc, labels

gist_desc, surf_desc, sift_desc, gabor_desc , lbp_desc, labels = load_data()

def generate_batch(query_img_features):
	train_gist = []
	train_sift = []
	train_surf = []
	for k,v in labels_index.items():
		start = v
		end = v + labels_sum[k]
		offset1 = random.randint(start,end-1)
		offset2 = offset1
		while(offset2 == offset1):
			offset2 = random.randint(start,end-1)
		offset3 = start
		while(offset3>=start and offset3<end):
			offset3 = random.randint(0,len(labels)-1)
		
		train_gist.append(np.concatenate((gist_desc[offset1] , gist_desc[offset2]),axis=0))
		train_sift.append(np.concatenate((sift_desc[offset1][0], sift_desc[offset2][0]),axis=0))
		train_surf.append(np.concatenate((surf_desc[offset1][0], surf_desc[offset2][0]),axis=0))

		train_gist.append(np.concatenate((gist_desc[offset1], gist_desc[offset3]),axis=0))
		train_sift.append(np.concatenate((sift_desc[offset1][0], sift_desc[offset3][0]),axis=0))
		train_surf.append(np.concatenate((surf_desc[offset1][0], surf_desc[offset3][0]),axis=0))
		
	return np.array(train_gist), np.array(train_sift), np.array(train_surf)


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

query_img_features = gen_query_features(misc.imread(query_img_file), bowDiction_sift, bowDiction_surf)
print(query_img_features)


# saver = tf.train.Saver()
# with tf.Session(graph=graph) as session:
# 	saver.restore(sess, "model.ckpt")
# 	print("Loaded Model")
	


# 	for i in len(gist_desc):

# 		a,b,c = generate_batch()
# 		# print("***************************",a.shape,b.shape,c.shape)
# 		# Prepare a dictionary telling the session where to feed the minibatch.
# 		feed_dict = {tf_train_gist: a, tf_train_sift: b,tf_train_surf: c}
# 		sim, l, _ = session.run([similarity], feed_dict=feed_dict)
	
# print("DONE")
