from matplotlib import pyplot as plt
from scipy import misc
import numpy as np
import cv2
import gist #https://github.com/yuichiroTCY/lear-gist-python
from skimage import feature
from sklearn.preprocessing import normalize

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

def build_gabor_filter():
	filters = []
	ksize = 31
	for theta in np.arange(0, np.pi, np.pi / 16):
		kern = cv2.getGaborKernel((ksize, ksize), 4.0, theta, 10.0, 1.0, 0, ktype=cv2.CV_32F)
		kern /= 1.5*kern.sum()
		filters.append(kern)
	return filters

def gabor(img, filters):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	accum = np.zeros_like(gray)
	for kern in filters:
		fimg = cv2.filter2D(gray, cv2.CV_8UC3, kern)
		np.maximum(accum, fimg, accum)
	return accum 


def local_binary_pattern(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	numPoints = 57
	radius = 8
	eps=1e-7
	# of the image, and then use the LBP representation
	# to build the histogram of patterns
	lbp = feature.local_binary_pattern(gray, numPoints,
		radius, method="uniform")
	(hist, _) = np.histogram(lbp.ravel(),
		bins=np.arange(0, numPoints + 3),
		range=(0, numPoints + 2))
 
	# normalize the histogram
	hist = hist.astype("float")
	hist /= (hist.sum() + eps)
 
	# return the histogram of Local Binary Patterns
	return hist

#displays currently edges image only, can't determine the edge direction histogram
def edge_direction_histogram(img):
	gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
	edges = cv2.Canny(gray,100,200)
	print(edges.shape)
	plt.subplot(121),plt.imshow(img,cmap = 'gray')
	plt.title('Original Image'), plt.xticks([]), plt.yticks([])
	plt.subplot(122),plt.imshow(edges,cmap = 'gray')
	plt.title('Edge Image'), plt.xticks([]), plt.yticks([])
	plt.show()


#img = cv2.imread('6.jpg')
img = misc.imread('6.jpg')
# d=sift(img)
print(img.shape)

# #gabor test src=https://cvtuts.wordpress.com/2014/04/27/gabor-filters-a-practical-overview/
# filters = build_gabor_filter()
# res1 = gabor(img, filters)
# print(res1.shape)
# cv2.imshow('result', res1)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

#local_binary_pattern src=http://www.pyimagesearch.com/2015/12/07/local-binary-patterns-with-python-opencv/
res2 = local_binary_pattern(img)

# #edge direction histogram
# edge_direction_histogram(img)