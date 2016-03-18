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
from sklearn.datasets import load_iris

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

import pickle

# General function to use in place of print (so can redirect output elsewhere or not print at all)
def display(*some_strings):

	no_print = True

	if not no_print:

		print(some_strings)

# Display that always prints out, should not be used inside loops or anything else that calls frequently
def cheap_display(*some_strings):

	print(some_strings)

def describe_array(input_array):

	display("Describing array")

	display(type(input_array))

	display(input_array.shape)

	line_break()

def line_break():
	display("---------------------------------------")

def save_classification_dict(classification_dict, filename, samples_per_class, max_total_images):

	with open(filename+"_"+str(samples_per_class)+"_"+str(max_total_images),'wb') as f:
		pickle.dump(classification_dict,f,pickle.HIGHEST_PROTOCOL)

def save_dictionary(dictionary, filename, samples_per_class, max_total_images, number_of_k_means_clusters):

	filename = filename + str(samples_per_class)+"_"+str(max_total_images)+"_"+str(number_of_k_means_clusters)

	f = open(filename, 'w')
	np.save(f, dictionary)
	f.close()

def save_descriptor_array(descriptor_array, filename, samples_per_class, max_total_images):

	filename = filename + "_" + str(samples_per_class)+"_"+str(max_total_images)

	f = open(filename, 'w')
	np.save(f, descriptor_array)
	f.close()

def save_classifier(classifier, filename, samples_per_class, max_total_images,number_of_k_means_clusters):

	filename = filename + "_" + str(samples_per_class)+"_"+str(max_total_images)+"_"+str(number_of_k_means_clusters)

	with open(filename,'wb') as f:
		pickle.dump(classifier,f,pickle.HIGHEST_PROTOCOL)

def save_encodings(encodings, filename, samples_per_class, max_total_images, number_of_k_means_clusters):

	filename = filename + "_" + str(samples_per_class)+"_"+str(max_total_images)+"_"+str(number_of_k_means_clusters)

	f = open(filename, 'w')
	np.save(f, encodings)
	f.close()

def load_numpy_object_file(filename):

	f = open(filename, 'r')
	return_object = np.load(f)
	f.close()

	return return_object

def load_pickle_object_file(filename):

	with open(filename,'rb') as f:
		return pickle.load(f)

def prepare_image(image):

	size = 300,300
 
	# resizes the image, preserves aspect ratio, resizes to be less than size
	image.thumbnail(size, Image.ANTIALIAS)

	#resize so all images are the same size
	image = image.resize(size)

	#print type(image)
	#display(image.size)

	image = array(image)
	image = color.rgb2gray(image)

	return image


#turns 3D daisy array into 2D array
def flatten_daisy(image):
	display("Flattening daisy")
	desc=daisy(image)
	#describe_array(desc)
	flat=desc[0,0,:]
	for i in range(1,67):
		for j in range(1,67):
			#print(len(desc[i,j,:]))
			flat=np.vstack((flat,desc[i,j,:]))
	describe_array(flat)
	return flat

def demo_iris():

	iris = load_iris()

	display("Iris data looks like: ")
	describe_array(iris.data)
	display(iris.data)

	display("Iris target looks like: ")
	describe_array(iris.target)
	display(iris.target)