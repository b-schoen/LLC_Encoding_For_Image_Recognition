import cv2

from utilities import describe_array

def do_cv_hog(image_pathfile):

	#help(cv2.HOGDescriptor())

	image = cv2.imread(image_pathfile)

	winSize = (24,24)
	blockSize = (16,16)
	blockStride = (8,8)
	cellSize = (8,8)
	nbins = 8
	derivAperture = 1
	winSigma = 4.
	histogramNormType = 0
	L2HysThreshold = 2.0000000000000001e-01
	gammaCorrection = 0
	nlevels = 64

	hog = cv2.HOGDescriptor(winSize,blockSize,blockStride,cellSize,nbins,derivAperture,winSigma,
	                        histogramNormType,L2HysThreshold,gammaCorrection,nlevels)

	#print(hog.getBlockHistogramSize)
	print(hog.getDescriptorSize())

	return hog.compute(image)

if __name__ == '__main__':

	image_directory = "Caltech-101/101_ObjectCategories/"

	test_image = image_directory+"accordion/image_0001.jpg"

	hog_result = do_cv_hog(test_image)

	describe_array(hog_result)

