from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
import cv2
import gist #https://github.com/yuichiroTCY/lear-gist-python
from six.moves import cPickle as pickle

def load_data():
	pickle_file = '/mnt/B8308B68308B2D06/CF/pickle_file_data2'
	with open(pickle_file, 'rb') as f:
		save = pickle.load(f)
		train_dataset = save['train_dataset']
		train_labels = save['train_labels']
		test_dataset = save['test_dataset']
		test_labels = save['test_labels']
		del save	# hint to help gc free up memory
		print('Training set', train_dataset.shape, len(train_labels))
		print('Test set', test_dataset.shape, len(test_labels))	
	return train_dataset.astype(np.uint8),train_labels

#3 histogram one for each color
def color_hist(img, histSize=256):
	hist = []
	hist[0] = cv2.calcHist([img],[0],None,[histSize],[0,256])
	hist[1] = cv2.calcHist([img],[1],None,[histSize],[0,256])
	hist[2] = cv2.calcHist([img],[2],None,[histSize],[0,256])
	#plt.plot(hist)
	#plt.xlim([0,256])
	#plt.show()
	return hist

#gist descriptor of size 960
def gist_descriptor(img):
	return gist.extract(img)

#sift descriptor of size number of (keypoint x 128)
def sift(img):
	plt.imshow(img)
	plt.show()
	gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
	sift = cv2.xfeatures2d.SIFT_create()
	(kp, desc) = sift.detectAndCompute(gray, None)
	# print(desc.shape)
	# print(len(kp))
	# im=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# plt.imshow(im)
	# plt.show()
	return kp, desc

#surf descriptor of size number of (keypoint x 128)
SURF_threashold = 500
def surf(img):
	surf = cv2.xfeatures2d.SURF_create(SURF_threashold)
	surf.setExtended(True)
	kp, desc = surf.detectAndCompute(img,None)
	# print(desc.shape)
	# print(len(kp))
	# im=cv2.drawKeypoints(img,kp,None,(255,0,0),4)
	# plt.imshow(im)
	# plt.show()
	return kp, desc

def save_vocabulary(BOWdict,name):
	f = open(name, 'wb')
	save = {
		name: BOWdict.getVocabulary()
	}
	pickle.dump(save, f, pickle.HIGHEST_PROTOCOL)
	f.close()

#######################Bag of words for SIFT#########################################
#for reference https://github.com/briansrls/SIFTBOW/blob/master/SIFTBOW.py
def bag_of_words_sift(data, dictionarySize=10):
	BOW = cv2.BOWKMeansTrainer(dictionarySize)
	for i in range(len(data)):
		try:
			kp, dsc = sift(data[i])
			BOW.add(dsc)
		except Exception: 
			print(i)
	
	print("Clustering")
	dictionary = BOW.cluster()
	print("Finish")

	sift2 = cv2.xfeatures2d.SIFT_create()
	bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
	bowDiction.setVocabulary(dictionary)
	save_vocabulary(bowDiction,"Sift_Voc")
	return bowDiction

def bow_feature_extract_sift(bowDiction, img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp,desc = sift(img)
    return bowDiction.compute(gray, kp, desc)

def load_sift_bow_diction(file):
	with open(file, 'rb') as f:
		save = pickle.load(f)
		vocabulary = save['Sift_Voc']
		del save	# hint to help gc free up memory
	sift2 = cv2.xfeatures2d.SIFT_create()
	bowDiction = cv2.BOWImgDescriptorExtractor(sift2, cv2.BFMatcher(cv2.NORM_L2))
	bowDiction.setVocabulary(vocabulary)
	return bowDiction

#######################Bag of words for SURF#########################################
def bag_of_words_surf(data, dictionarySize=10):
	BOW = cv2.BOWKMeansTrainer(dictionarySize)
	x=0
	for i in range(len(data)):
		try:
			kp, dsc = surf(data[i])
			BOW.add(dsc)
		except Exception: 
			print(i)
			x+=1
	print("--------------------------------->",x)
	print("Clustering")
	dictionary = BOW.cluster()
	print("Finish")

	surf2 = cv2.xfeatures2d.SURF_create(SURF_threashold)
	surf2.setExtended(True)
	bowDiction = cv2.BOWImgDescriptorExtractor(surf2, cv2.BFMatcher(cv2.NORM_L2))
	bowDiction.setVocabulary(dictionary)
	save_vocabulary(bowDiction,"Surf_Voc")
	return bowDiction

def bow_feature_extract_surf(bowDiction, img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    kp,desc = surf(img)
    return bowDiction.compute(gray, kp, desc)

def load_surf_bow_diction(file):
	with open(file, 'rb') as f:
		save = pickle.load(f)
		vocabulary = save[file]
		del save	# hint to help gc free up memory
	surf = cv2.xfeatures2d.SURF_create(SURF_threashold)
	surf.setExtended(True)
	bowDiction = cv2.BOWImgDescriptorExtractor(surf, cv2.BFMatcher(cv2.NORM_L2))
	bowDiction.setVocabulary(vocabulary)
	return bowDiction
########################################################################################

data, label = load_data()
data, label = data[:10000], label[:10000]

BOWdict = bag_of_words_sift(data)
BOWdict = load_sift_bow_diction("Surf_Voc")


for i in range(10):
	d = bow_feature_extract_sift(BOWdict,data[i])
	print(d.shape,label[i])
	print(d)
	print("__________________________________________________________")


print("Done")