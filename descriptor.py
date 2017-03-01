from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
import cv2
import gist #https://github.com/yuichiroTCY/lear-gist-python

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

def sift(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	#sift = cv2.xfeatures2d.SIFT_create(contrastThreshold=0.05,edgeThreshold=500)
	sift = cv2.xfeatures2d.SIFT_create()
	(kp, desc) = sift.detectAndCompute(gray, None)
	# print(desc.shape)
	# print(len(kp))
	# im=cv2.drawKeypoints(gray,kp,None,flags=cv2.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)
	# plt.imshow(im)
	# plt.show()
	return desc


def surf(img):
	surf = cv2.xfeatures2d.SURF_create(500)
	kp, desc = surf.detectAndCompute(img,None)
	# print(desc.shape)
	# print(len(kp))
	# im=cv2.drawKeypoints(img,kp,None,(255,0,0),4)
	# plt.imshow(im)
	# plt.show()
	return desc

#img = cv2.imread('6.jpg')
img = misc.imread('6.jpg')
d=sift(img)
print(d.shape)