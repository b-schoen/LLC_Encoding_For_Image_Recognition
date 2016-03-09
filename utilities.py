#images and features
from skimage.feature import hog, daisy
from skimage import data, color, exposure

#kmeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import SVC

#numpy
from numpy import array
import numpy as np

#time
from time import time

#image
import Image

#inspect functions
import inspect

#math (real and complex)
import math
import cmath

def describe_array(input_array):

	print("Describing array")

	print(type(input_array))

	#print(np.dtype(input_array[0,0]))

	print(input_array.shape)

	line_break()

def line_break():
	print("---------------------------------------")


def save_dictionary(dictionary, filename, samples_per_class, max_total_images, number_of_k_means_clusters):

	filename = filename + "_" + str(samples_per_class)+"_"+str(max_total_images)+"_"+str(number_of_k_means_clusters)

	f = open(filename, 'w')
	np.save(f, dictionary)
	f.close()

def save_descriptor_array(descriptor_array, filename, samples_per_class, max_total_images):

	filename = filename + "_" + str(samples_per_class)+"_"+str(max_total_images)

	f = open(filename, 'w')
	np.save(f, descriptor_array)
	f.close()

def load_file(filename):

	f = open(filename, 'r')
	return_object = np.load(f)
	f.close()

	return return_object

def prepare_image(image):

	size = 300,300
 
	# resizes the image, preserves aspect ratio, resizes to be less than size
	image.thumbnail(size, Image.ANTIALIAS)

	#resize so all images are the same size
	image = image.resize(size)

	#print type(image)
	print(image.size)

	image = array(image)
	image = color.rgb2gray(image)

	return image


#turns 3D daisy array into 2D array
def flatten_daisy(image):
	print("Flattening daisy")
	desc=daisy(image)
	#describe_array(desc)
	flat=desc[0,0,:]
	for i in range(1,67):
		for j in range(1,67):
			#print(len(desc[i,j,:]))
			flat=np.vstack((flat,desc[i,j,:]))
	describe_array(flat)
	return flat