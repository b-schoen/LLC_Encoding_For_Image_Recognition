# -*- coding: utf-8 -*-

#TODO: Dig into modified_skimage.py to directly use histograms mid hog for pooling / normalization (for SPM)
#TODO: Resolve all other TODOs

#matlab plotting
#import matplotlib.pyplot as plt

print("Importing packages...")

#images and features
from skimage.feature import hog, daisy
from skimage import data, color, exposure

#modified skimage functions to return what we want
from modified_skimage import modified_hog

#kmeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.preprocessing import normalize
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris
from sklearn.ensemble import BaggingClassifier

#numpy
from numpy import array
import numpy as np

#os - for file navigation
import os

#time
from time import time

#image
import Image

#inspect functions
import inspect

#math (real and complex)
import math
import cmath

# functions from utilities.py
from utilities import display
from utilities import cheap_display
from utilities import describe_array
from utilities import line_break
from utilities import save_classification_dict
from utilities import save_dictionary
from utilities import save_descriptor_array
from utilities import save_encodings
from utilities import save_classifier
from utilities import load_numpy_object_file
from utilities import load_pickle_object_file
from utilities import prepare_image
from utilities import flatten_daisy
from utilities import demo_iris

# functions from SPM.py
from SPM import pooling

def do_k_means(input_data, clusters):

	cheap_display("Doing k means...")
	cheap_display(input_data.shape)

	#batch = input_data.shape[0]/100
	batch_factor = 10
	batch = batch_factor*clusters
	batch = int(batch)

	#normalize data
	normalize(input_data,axis=0,copy=False)

	#scale data
	#input_data = scale(input_data)

	#get samples and features
	n_samples, n_features = input_data.shape

	#kmeans_function = KMeans(init='k-means++', n_clusters=clusters, n_init=1, verbose=1)

	# use MiniBatchKMeans for faster performance (smaller batches means managable data size)
	#	comes at the cost of worse accuracy
	kmeans_function = MiniBatchKMeans(batch_size=batch, init='k-means++', n_clusters=clusters, n_init=10, max_no_improvement=10, verbose=1, reassignment_ratio=0)

	return bench_k_means(kmeans_function, name="k-means++", data=input_data)

def bench_k_means(estimator, name, data):
    t0 = time()
    km=estimator.fit(data)

    cheap_display("Done k means")

    return km.cluster_centers_

# Raveled version of the hog descriptors
# From Dalal and Triggs:
#		larger code vector implicitly encodes spatial information relative to the position window
def get_hog_descriptor(image_file):

	orientations = 8
	pixels_per_cell = (8,8)
	cells_per_block = (4,4)

	image = Image.open(image_file)
	image = prepare_image(image)

	feature_descriptor = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

	descriptors = [feature_descriptor]

	return descriptors

# From the skimage comments: 
#		We refer to the normalised block descriptors as Histogram of Oriented Gradient (HOG) descriptors.
def get_modified_hog_descriptor(image_file):

	#NOTE: Parameters are correct and result in same #dims as HOG descriptors used in paper

	orientations = 8
	pixels_per_cell = (8,8)
	cells_per_block = (4,4)
	dimension = orientations*cells_per_block[0]*cells_per_block[1]

	image = Image.open(image_file)
	image = prepare_image(image)

	normalised_blocks = modified_hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block)

	descriptors = []

	#NOTE: normalized blocks (x,y) are still (2,2,9), because have histograms for each cell in block (so just raveling and seeing if it works)
	#NOTE: 36 = 2 * 2 * 9 => total size of these is based on cells in block and orientations
	descriptors=np.empty((normalised_blocks.shape[0],normalised_blocks.shape[1],dimension))
	for x in xrange(normalised_blocks.shape[0]):
		for y in xrange(normalised_blocks.shape[1]):
			#descriptors[x,y]=normalised_blocks[x,y].ravel()
			descriptors[x,y]=(normalised_blocks[x,y].ravel()).astype(np.float32)

	return descriptors

def compare_hog_descriptors(image_file):

	#NOTE: dimension of hog = product of all array dimensions of modified hog

	display("Hog is: ")

	image_hog = get_hog_descriptor(image_file)

	describe_array(image_hog)

	display("Modified hog is: ")

	image_modified_hog = get_modified_hog_descriptor(image_file)

	for descriptor in image_modified_hog:

		describe_array(descriptor)

def get_spm_parameters(test_image):

	im0=Image.open(test_image)
	im0=prepare_image(im0)
	sub, desc=spm_subregions(get_modified_hog_descriptor(test_image))

	return sub, desc

def spm_subregions(desc, partitions=4):

	h = desc.shape[0]
	w = desc.shape[1]
	w=w/partitions
	h=h/partitions

	descriptor_list=[]
	sr=()
	for i in range(0,partitions): 
		for j in range(0,partitions):
			quart = desc[(i*h):(i+1)*h,(j*w):(j+1)*w]
			q=[]
			for x in xrange(quart.shape[0]):
				for y in xrange(quart.shape[1]):
					q.append(quart[x,y])
			sr=sr+(q,)
	for x in xrange(desc.shape[0]):
				for y in xrange(desc.shape[1]):
					descriptor_list.append(desc[x,y])
	return sr, descriptor_list

def get_spm_encodings(spm_subregions,neigh,dictionary, number_of_k_means_clusters):

	cheap_display("Getting SPM encodings...")

	images_processed = 0

	#print(spm_subregions.shape)
	spm_encodings=[]
	for img in spm_subregions:
		#print(img.shape)
		pooled_1=[]
		pooled_2=[]
		for sub in img:
			#describe_array(sub)
			pooled_1.append(pooling(get_encodings_list(sub,neigh,dictionary,number_of_k_means_clusters), 'max'))
		#print(pooled_1)
		for i in range(0,4):
			pooled_2.append(pooling(pooled_1[(4*i):(4*i)+3], 'max'))
		#	#rint(i)
		#pooling_result=pooling(pooled_1,'max')
		#encoding = normalize(pooling_result, 'l_2')
		encoding=pooling(pooled_2,'max')
		spm_encodings.append(encoding)
		#print(len(encoding))
		#print('encoded an image')

		images_processed = images_processed + 1
		cheap_display("Encoded ",images_processed," images so far")

	spm_encodings_array = np.asarray(spm_encodings)

	cheap_display("SPM encodings: ")
	cheap_display(spm_encodings_array.shape)

	return spm_encodings_array

def run_on_all_images(process_image, image_directory, images_to_sample_per_class, number_to_stop_at, calc_descriptors, testing=False, descriptor_only=False):

	cheap_display("Running on all images...")

	#TODO: Make sure these are being run on images in the correct order

	#specify generic form of return data list
	return_data_list = []
	subregions_list = ()

	classification_list = []
	classification_dict = {}
	class_count = 0

	#number of images processed
	images_processed = 0

	#number of images to sample per class
	images_sampled_in_class = 0

	display(image_directory)


	# For every class
	class_files = os.listdir(image_directory)

	for filename in class_files:

		display(filename)

		current_class = filename

		#classification_dict[current_class] = class_count
		classification_dict[class_count] = current_class

		filename = image_directory +"/" + filename

		images_sampled_in_class = 0


		# For every instance of that class
		class_instance_files = os.listdir(filename)

		# If testing, reverse the list so we start from the end, that way training and testing use the beginning and end of list respectively
		if(testing):
			class_instance_files.reverse()

		for image_file in class_instance_files:

			if(images_sampled_in_class < images_to_sample_per_class):

				#display(image_file)

				image_file = filename +"/" + image_file

				if(calc_descriptors):

					descriptors = process_image(image_file)

					subregions, descriptor = spm_subregions(descriptors)

					subregions_list = subregions_list + (subregions,)

					# if we eventually want encodings
					if not descriptor_only:

						for description in descriptor:
							return_data_list.append(description)

						#log classification of this images
						#	Appending outside of the descriptor iterator loop is fine, as descriptors are eventually pooled
						classification_list.append(class_count)

					else:

						for description in descriptor:
							return_data_list.append(description)
							classification_list.append(class_count)

					#return if reached max images total to process
					images_processed = images_processed + 1
					cheap_display("Processed ",str(images_processed)," images so far")
					if(images_processed >= number_to_stop_at):

						raise UserError("Return case not implemented")

						return return_data_list

				line_break()
				line_break()

			images_sampled_in_class = images_sampled_in_class+1

		#done with this class
		class_count = class_count + 1


	display(classification_list)
	display(classification_dict)

	# convert return data list to vstacked array all at once, to use less copies
	if(calc_descriptors):
		cheap_display("Stacking descriptors...")

		return_array = np.vstack(return_data_list)
		cheap_display(return_array.shape)

		cheap_display("Making subregions list an array...")
		subregions_array = np.asarray(subregions_list)

		# convert classification list to target array
		target_array = np.asarray(classification_list)

	else:
		return_array = np.empty([1,1])
		subregions_array = np.empty([1,1])
		target_array = np.empty([1,1])

	return subregions_array, return_array, target_array, classification_dict

def find_local_basis(descriptor, neigh, dictionary):
	ind=neigh.kneighbors(descriptor)[1]
	B=dictionary[ind,:]
	return ind, np.squeeze(B)

def get_descriptor_encoding(descriptor, neigh, dictionary, number_of_k_means_clusters):

	xi = descriptor.reshape(1,-1)
	ind, bi = find_local_basis(xi, neigh, dictionary)
	ci = np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[0] #k-length encoding

	ci = np.transpose(ci)

	enc = np.zeros(number_of_k_means_clusters)

	for i in xrange(len(ind)):
		enc[ind[[i]]]=ci[i]

	return enc

def get_encodings_list(descriptor_array, neigh, dictionary, number_of_k_means_clusters):

	encoding_list = []

	#Should be column because descriptors are stacked vertically
	for descriptor in descriptor_array:

		encoding_list.append(np.transpose(get_descriptor_encoding(descriptor, neigh, dictionary, number_of_k_means_clusters)))

	return encoding_list

def train_classifier(training_data, target):

	#NOTE: For final SPM, these will actually be normalized, thus scaled and normalized should give accurate results
	#		This is probably the problem now, and not worth normalizing without SPM

	#TODO: Can still using bagging to optimize

	cheap_display("Training classifier...")

	cheap_display("Training data: ")
	cheap_display(training_data.shape)

	cheap_display("Target: ")
	cheap_display(target.shape)

	#This is just reformatting for numpy's sake, not actually 'changing' shape
	target.reshape(training_data.shape[0],1)

	penalty_parameter = 1

	linear_svc = LinearSVC(C=penalty_parameter,verbose=1)
	
	cheap_display("Fitting classifier...")
	linear_svc.fit(training_data, target)

	return linear_svc

def class_from_target(classification_data, classification_dict):

	result_list = []

	for classification in np.nditer(classification_data):

		#display(type(classification))
		#display(type(classification_data))
		#display(type(classification_dict))

		classification_key = int(classification)

		class_prediction = classification_dict[classification_key]

		display("Class prediction is: ")
		display(class_prediction)

		result_list.append(class_prediction)

	return result_list

def get_predicted_class(classifier, encoding, classification_dict):

	# get prediction
	#NOTE: Using only for now while passing one in at a time, can probably hugely speed up by doing all at once
	#		Can probably just put in raw mulitple encodings, will be appropriate shape
	encoding = encoding.reshape(1,-1)

	prediction = classifier.predict(encoding)

	prediction = int(prediction[0])

	class_instance = classification_dict[prediction]

	return class_instance

def get_actual_class(encoding_index, target_array, classification_dict):

	 classification_key = target_array[encoding_index]

	 class_instance = classification_dict[classification_key]

	 return class_instance

def get_accuracy(classifier, testing_encodings, testing_target_array, testing_classification_dict):


	hit_count = 0
	total_count = 0

	for i in range(testing_encodings.shape[0]):

		predicted_class_instance = get_predicted_class(classifier, testing_encodings[i], testing_classification_dict)

		actual_class_instance = get_actual_class(i, testing_target_array, testing_classification_dict)

		cheap_display(actual_class_instance, predicted_class_instance)

		if(predicted_class_instance == actual_class_instance):

			hit_count = hit_count + 1

		total_count = total_count + 1

	cheap_display("Total hits: ",hit_count)
	cheap_display("Total images: ",total_count)

	accuracy = float(hit_count) / float(total_count)

	return accuracy

def main():

	#TODO: Generates taking in these parameters

	# What to use ------------------------------

	#NOTE: Changing Hog and modified hog not implimented in saved file names yet, so will overwrite each other
	use_modified_hog = True

	# ------------------------------------------

	# What to generate --------------------------

	training_generate_descriptors = False
	testing_generate_descriptors = False #Stays false when testing # is constant

	training_generate_dictionary = False

	training_generate_encodings = False
	testing_generate_encodings = False

	train_encodings_based_classifier = True

	# -------------------------------------------


	# Generate training -----------------------------------------------------------------------------------

	# Get descriptors
	cheap_display("Getting training descriptors...")

	#NOTE: both classification dicts match
	if use_modified_hog:
		training_subregions, training_descriptors, training_target_array, training_classification_dict = run_on_all_images(get_modified_hog_descriptor, image_directory, training_samples_per_class, max_total_images, training_generate_descriptors)
		descriptor_file_label_training = 'modified_hog_descriptors_training'
	else:
		training_subregions, training_descriptors, training_target_array, training_classification_dict = run_on_all_images(get_hog_descriptor, image_directory, training_samples_per_class, max_total_images, training_generate_descriptors)
		descriptor_file_label_training = 'hog_descriptors_training'

	#save arrays so don't have to recalculate each time
	if(training_generate_descriptors):

		save_descriptor_array(training_descriptors, descriptor_file_label_training, training_samples_per_class, max_total_images)
		save_classification_dict(training_classification_dict, 'training_classification_dict', training_samples_per_class, max_total_images)
		save_descriptor_array(training_subregions, 'SPM_'+descriptor_file_label_training, training_samples_per_class, max_total_images)
		save_descriptor_array(training_target_array, 'training_target_'+descriptor_file_label_training, training_samples_per_class, max_total_images)

	else:

		if(training_generate_encodings):

			#load previously-made array of descriptors
			cheap_display("Loading training descriptors...")
			training_descriptors = load_numpy_object_file(descriptor_file_label_training+"_"+str(training_samples_per_class)+'_'+str(max_total_images))

			cheap_display("Loading training SPM subregions...")
			training_subregions = load_numpy_object_file('SPM_'+descriptor_file_label_training+"_"+str(training_samples_per_class)+'_'+str(max_total_images))

		else:

			training_descriptors = np.empty([1,1])
			training_subregions = np.empty([1,1])

		cheap_display("Loading training classification dict...")
		training_classification_dict = load_pickle_object_file('training_classification_dict'+"_"+str(training_samples_per_class)+"_"+str(max_total_images))

		cheap_display("Loading training target array...")
		training_target_array = load_numpy_object_file('training_target_'+descriptor_file_label_training+"_"+str(training_samples_per_class)+'_'+str(max_total_images))

	print("Training descriptors shape: ")
	print(training_descriptors.shape)

	print("Training subregions shape: ")
	print(training_subregions.shape)

	# Scale data
	data_scaler = StandardScaler()
	data_scaler.fit(training_descriptors)
	training_descriptors = data_scaler.transform(training_descriptors)

	# Get dictionary

	#run only if haven't already generated descriptors 
	cheap_display("Getting dictionary...")
	if(training_generate_dictionary):

		dictionary = do_k_means(training_descriptors, number_of_k_means_clusters)

		save_dictionary(dictionary, 'dictionary', training_samples_per_class, max_total_images, number_of_k_means_clusters)
	else:

		#load previously-made dictionary
		cheap_display("Loading dictionary...")
		dictionary = load_numpy_object_file('dictionary'+"_"+str(training_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))

	# Fit dictionary
	cheap_display("Fitting dictionary...")
	neigh = NearestNeighbors(n_neighbors=k_neighbors)
	cheap_display(dictionary.shape)
	neigh.fit(dictionary)

	# Get encodings

	#run only if haven't already generated encodings
	cheap_display("Getting training encodings...")
	if(training_generate_encodings):

		training_encodings = get_spm_encodings(training_subregions, neigh, dictionary, number_of_k_means_clusters)

		save_encodings(training_encodings, "encodings_training", training_samples_per_class, max_total_images, number_of_k_means_clusters)

	else:

		cheap_display("Loading training encodings...")
		training_encodings = load_numpy_object_file("encodings_training_"+str(training_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))

	# Delete large unused objects
	del training_descriptors
	del training_subregions

	#------------------------------------------------------------------------------------------------------------


	


	# Generate for testing ---------------------------------------------------------------------------------

	# Generate descriptors
	cheap_display("Getting testing descriptors...")
	if use_modified_hog:
		testing_subregions, testing_descriptors,  testing_target_array, testing_classification_dict= run_on_all_images(get_modified_hog_descriptor, image_directory, testing_samples_per_class, max_total_images, testing_generate_descriptors, testing=True)
		descriptor_file_label_testing = 'modified_hog_descriptors_testing'
	else:
		testing_subregions, testing_descriptors,  testing_target_array, testing_classification_dict= run_on_all_images(get_hog_descriptor, image_directory, testing_samples_per_class, max_total_images, testing_generate_descriptors, testing=True)
		descriptor_file_label_testing = 'hog_descriptors_testing'

	#save arrays so don't have to recalculate each time
	if(testing_generate_descriptors):

		save_descriptor_array(testing_descriptors, descriptor_file_label_testing, testing_samples_per_class, max_total_images)
		save_classification_dict(testing_classification_dict, 'testing_classification_dict', testing_samples_per_class, max_total_images)
		save_descriptor_array(testing_subregions, 'SPM_'+descriptor_file_label_testing, testing_samples_per_class, max_total_images)
		save_descriptor_array(testing_target_array, 'testing_target_'+descriptor_file_label_testing, testing_samples_per_class, max_total_images)

	else:

		if(testing_generate_encodings):

			#load previously-made array of descriptors
			cheap_display("Loading testing descriptors...")
			testing_descriptors = load_numpy_object_file(descriptor_file_label_testing+"_"+str(testing_samples_per_class)+'_'+str(max_total_images))

			cheap_display("Loading testing SPM subregions...")
			testing_subregions = load_numpy_object_file('SPM_'+descriptor_file_label_testing+"_"+str(testing_samples_per_class)+'_'+str(max_total_images))

		else:

			testing_descriptors = np.empty([1,1])
			testing_subregions = np.empty([1,1])

		cheap_display("Loading testing classification dict...")
		testing_classification_dict = load_pickle_object_file('testing_classification_dict'+"_"+str(testing_samples_per_class)+"_"+str(max_total_images))

		cheap_display("Loading training target array...")
		testing_target_array = load_numpy_object_file('testing_target_'+descriptor_file_label_testing+"_"+str(testing_samples_per_class)+'_'+str(max_total_images))

	# Get encodings

	#run only if haven't already generated encodings
	cheap_display("Getting testing encodings...")
	if(testing_generate_encodings):

		testing_encodings = get_spm_encodings(testing_subregions, neigh, dictionary, number_of_k_means_clusters)

		save_encodings(testing_encodings, "encodings_testing", testing_samples_per_class, max_total_images, number_of_k_means_clusters)

	else:

		cheap_display("Loading testing encodings...")
		testing_encodings = load_numpy_object_file("encodings_testing_"+str(testing_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))

	# Delete large unused objects
	del testing_descriptors
	del testing_subregions


	# --------------------------------------------------------------------------------------------------------



	# Train classifier and predict results -----------------------------------------------------------------

	#TODO: Is classification better doing or not doing transpose AFTER encodings (training) are done?

	# assert that the testing and training classificaiton dictionaries match
	assert (training_classification_dict == testing_classification_dict)


	# normalize encodings for classifier
	normalize(training_encodings,axis=0,copy=False)
	normalize(testing_encodings,axis=0,copy=False)

	# scale encodings
	training_encodings = data_scaler.transform(training_encodings)
	testing_encodings = data_scaler.transform(testing_encodings)
	#testing_encodings = scale(testing_encodings)
	

	if(train_encodings_based_classifier):
	
		cheap_display("Training encoding based classifier: ")
		encoding_based_classifier = train_classifier(training_encodings, training_target_array)
		save_classifier(encoding_based_classifier, "classifier_encoding_based", training_samples_per_class, max_total_images,number_of_k_means_clusters)

	else:

		cheap_display("Loading encoding based classifier: ")
		encoding_based_classifier = load_pickle_object_file("classifier_encoding_based"+ "_" + str(training_samples_per_class)+"_"+str(max_total_images)+"_"+str(number_of_k_means_clusters))


	cheap_display("Getting accuracy for encoding based classifier...")
	encoding_based_accuracy_different = get_accuracy(encoding_based_classifier, testing_encodings, testing_target_array, testing_classification_dict)
	encoding_based_accuracy_same = get_accuracy(encoding_based_classifier, training_encodings, training_target_array, testing_classification_dict)
	cheap_display("Accuracy for encoding based classifier (different) is: ", encoding_based_accuracy_different)
	cheap_display("Accuracy for encoding based classifier (same) is: ", encoding_based_accuracy_same)

	# -------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

	# Datasets

	image_directory = "Caltech-101/101_ObjectCategories"
	test_image = "test_image.jpg"

	# Parameters

	k_neighbors = 5
	total_classes = 102
	# vary from 5,...,30
	training_samples_per_class = 2
	# keep at 30 for all trials		 					
	testing_samples_per_class = 1					
	max_total_images = 10000
	number_of_k_means_clusters = 128					#base of codebook

	main()
	
