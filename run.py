#TODO: Resolve all other TODOs

#matlab plotting
#import matplotlib.pyplot as plt

#images and features
from skimage.feature import hog, daisy
from skimage import data, color, exposure

#modified skimage functions to return what we want
from modified_skimage import modified_hog

#kmeans
from sklearn import metrics
from sklearn.cluster import KMeans
from sklearn.datasets import load_digits
from sklearn.decomposition import PCA
from sklearn.preprocessing import scale
from sklearn.neighbors import NearestNeighbors
from sklearn.svm import LinearSVC
from sklearn.datasets import load_iris

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
from utilities import save_dictionary
from utilities import save_descriptor_array
from utilities import save_encodings
from utilities import load_file
from utilities import prepare_image
from utilities import flatten_daisy
from utilities import demo_iris

# functions from SPM.py
from SPM import pooling
from SPM import normalize

def do_k_means(input_data, clusters):

	cheap_display("Doing k means...")

	#scale data
	scaled_data = scale(input_data)

	#get samples and features
	n_samples, n_features = scaled_data.shape

	return bench_k_means(KMeans(init='k-means++', n_clusters=clusters, n_init=10, verbose=1),
              name="k-means++", data=scaled_data)

def bench_k_means(estimator, name, data):
    t0 = time()
    km=estimator.fit(data)

    cheap_display("Done k means")

    return km.cluster_centers_

    #for additional info
    '''print('% 9s   %.2fs    %i   %.3f   %.3f   %.3f   %.3f   %.3f    %.3f'
          % (name, (time() - t0), estimator.inertia_,
             metrics.homogeneity_score(labels, estimator.labels_),
             metrics.completeness_score(labels, estimator.labels_),
             metrics.v_measure_score(labels, estimator.labels_),
             metrics.adjusted_rand_score(labels, estimator.labels_),
             metrics.adjusted_mutual_info_score(labels,  estimator.labels_),
             metrics.silhouette_score(data, estimator.labels_,
                                      metric='euclidean',
                                      sample_size=sample_size)))'''

def get_hog_descriptor(image_file):

	orientations = 9
	pixels_per_cell = (8,8)
	cells_per_block = (2,2)
	return_as_feature_vector = True

	image = Image.open(image_file)
	image = prepare_image(image)

	(feature_descriptor, hog_image) = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=True)

	#return hog_image

	#display(feature_descriptor)
	return feature_descriptor

def run_on_all_images(process_image, image_directory, images_to_sample_per_class, number_to_stop_at, calc_descriptors, testing=False):

	cheap_display("Running on all images...")

	#TODO: Make sure these are being run on images in the correct order

	#specify generic form of return data list
	return_data_list = []

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

					descriptor = process_image(image_file)

					return_data_list.append(descriptor)

					#return if reached max images total to process
					images_processed = images_processed + 1
					cheap_display("Processed ",str(images_processed)," images so far")
					if(images_processed >= number_to_stop_at):

						raise UserError("Return case not implemented")

						return return_data_list

				#log classification of this images
				classification_list.append(class_count)

				line_break()
				line_break()

			images_sampled_in_class = images_sampled_in_class+1

		#done with this class
		class_count = class_count + 1


	display(classification_list)
	display(classification_dict)


	# convert classification list to target array
	target_array = np.asarray(classification_list)

	# convert return data list to vstacked array all at once, to use less copies
	if(calc_descriptors):
		return_array = np.vstack(return_data_list)
		describe_array(return_array)
	else:
		return_array = np.empty([1,1])

	return return_array, target_array, classification_dict

def optimize_codebook(codebook, batch_of_descriptors, neigh):

	## Two hashes imply psuedocode from paper

	##input: B init ∈ R D×M , X ∈ R D×N , λ, σ

	# set params -------------------------------------------------------------------------

	# from paper
	lambda_value = float(500)
	sigma_value = float(100)

	# from paper / our parameters
	#codebook_size = codebook.shape[0]

	display("Codebook has shape: ", codebook.shape)

	display("Batch of descriptors has shape: ", batch_of_descriptors.shape)

	# D - dimension of local descriptors
	D = batch_of_descriptors.shape[1]

	# N - number of descriptors in this batch of the online learning algorithm
	#		-	This is why X is DxN, because it's an array of N descriptor vectors
	N = batch_of_descriptors.shape[0]

	# M - number of entries (vectors) in codebook
	M = codebook.shape[0]

	display("D is: ", D)
	display("N is: ", N)
	display("M is: ", M)

	# --------------------------------------------------------------------------------------

 	##B ← B init .
 	optimized_codebook = np.copy(codebook)

 	##for i = 1 to N do
 	for i in range(1,N):

 		display("-------------------------------------------------------------------------------------")
 		display(i)

 		x_i = batch_of_descriptors[i,:]
 		display(x_i.shape)

		##	d ← 1 × M zero vector, 
		d = np.zeros((1,M))

		##{locality constraint parameter} ------------------

		display("{locality constraint parameter} ------------------")

		##for j = 1 to M do
		for j in range(0,M):

			##	d j ← exp −1 − x i − b j 2 /σ .
			d_j = np.linalg.norm(x_i-codebook[j,:])
			d_j = d_j*d_j
			d_j = -d_j
			d_j =  d_j / sigma_value

			#TODO: Is this natural log or 1/e^x?
			#TODO: Plain exp is definitely wrong because it ignores the -1, but I don't see how they're getting the -1 from their paper
			#		And why is distance negative?
			d_j = math.exp(d_j)

			d[0,j] =  d_j

		##end for

		##d ← normalize (0,1] (d) 
		#TODO: Correct type of normalization?
		d = normalize(d,"l_2")

		##{coding} -----------------------------------------

		display("{coding} -----------------------------------------")

		##c i ← argmax c ||x i − Bc|| 2 + λ||d c|| 2 s.t. 1 c = 1. 
		#TODO: Assuming coding here can just use K-NN to approx like we did earlier
		c_i = get_encoding(i, batch_of_descriptors, neigh, codebook)
		display(c_i)

		##{remove bias} ------------------------------------

		display("{remove bias} ------------------------------------")

		##id ← {j|abs c i (j) > 0.01}
		id_found = False
		j=0
		while(not id_found):
			if(math.fabs(c_i[0,j]) > 0.01 ):
				id_value = j
				id_found = True
			j = j + 1

		##B i ← B(:, id)
		#TODO: What does this need to be size wise? Their notation doesn't mean slicing the same way
		B_i = codebook[id_value,:]
		display(B_i.shape)

		##c i ← argmax c ||x i − B i c|| 2 s.t. j c(j) = 1.
		#TODO: This is without that constraint
		tilde_c_i = np.linalg.lstsq(np.transpose(B_i),np.transpose(x_i))[0]

		##{update basis} -----------------------------------

		display("{update basis} -----------------------------------")

		##ΔB i ←= −2 c i (x i − B i  ̃ c i )
		#TODO: How do I enforce that constraint?
		#delta_B_i =  -2*tilde_c_i*(x_i-B_i*tilde_c_i)

		##μ ← 1/i
		#mu = 1.0 / i

		##B i ← B i − μΔB i /| ̃c i | 2

		##B(:, id) ← proj(B i ).


	##output: B

	return "in progress"

def find_local_basis(descriptor, neigh, dictionary):
	ind=neigh.kneighbors(descriptor)[1]
	B=dictionary[ind,:]
	return np.squeeze(B)

def get_encoding(index_of_descriptor, descriptor_array, neigh, dictionary):

	display("Getting encoding...")

	#testing encoding on 1 descriptor at index_of_descriptor
	xi = descriptor_array[index_of_descriptor].reshape(1,-1)
	bi = find_local_basis(xi, neigh, dictionary)
	ci = np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[0] #k-length encoding
	err= np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[1]

	#display results
	'''display("Descriptor for x_i: ")
	describe_array(xi)
	display(xi)

	display("Base for x_i: ")
	describe_array(bi)
	display(bi)

	display("Code for x_i: ")
	describe_array(ci)
	display(ci)

	display("Error for x_i: ")
	describe_array(err)
	display(err)'''

	#REMOVE
	ci = np.transpose(ci)

	return ci

def get_descriptor_encoding(descriptor, neigh, dictionary):

	xi = descriptor.reshape(1,-1)
	bi = find_local_basis(xi, neigh, dictionary)
	ci = np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[0] #k-length encoding

	#REMOVE
	ci = np.transpose(ci)

	return ci

def get_encodings_array(descriptor_array, neigh, dictionary):

	encoding_list = []

	#TODO: Is it rows instead?
	for column in descriptor_array:

		encoding_list.append(get_descriptor_encoding(column, neigh, dictionary))

	return np.squeeze(np.asarray(encoding_list))

def train_classifier(total_classes, samples_per_class, training_data, target):

	cheap_display("Target looks like: ")

	describe_array(target)

	target.reshape(total_classes*samples_per_class,1)

	linear_svc = LinearSVC()
	
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
	prediction = int(classifier.predict(encoding.reshape(1,-1))[0])

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

		if(predicted_class_instance == actual_class_instance):

			hit_count = hit_count + 1

		total_count = total_count + 1

	accuracy = float(hit_count) / float(total_count)

	return accuracy


def main():

	#TODO: Generates taking in these parameters

	generate_descriptors = False
	generate_dictionary = True
	generate_encodings = True



	#create and save new array of descriptors -------------------------------------------------------

	#get descriptors
	#NOTE: both classification dicts match
	training_descriptor_array, training_target_array, training_classification_dict = run_on_all_images(get_hog_descriptor, image_directory, training_samples_per_class, max_total_images, generate_descriptors)
	testing_descriptor_array,  testing_target_array, testing_classification_dict= run_on_all_images(get_hog_descriptor, image_directory, testing_samples_per_class, max_total_images, generate_descriptors, testing=True)

	#save arrays so don't have to recalculate each time
	if(generate_descriptors):

		save_descriptor_array(training_descriptor_array, 'hog_descriptors_training', training_samples_per_class, max_total_images)
		save_descriptor_array(testing_descriptor_array, 'hog_descriptors_testing', testing_samples_per_class, max_total_images)

	#------------------------------------------------------------------------------------------------------------



	#create & save new dictionary -------------------------------------------------------------------------------

	#load previously-made array of descriptors
	training_descriptor_array = load_file('hog_descriptors_training_'+str(training_samples_per_class)+'_'+str(max_total_images))
	testing_descriptor_array = load_file('hog_descriptors_testing_'+str(testing_samples_per_class)+'_'+str(max_total_images))

	#run only if haven't already generated descriptors 
	if(generate_dictionary):

		dictionary = do_k_means((training_descriptor_array), number_of_k_means_clusters)

		save_dictionary(dictionary, 'dictionary', training_samples_per_class, max_total_images, number_of_k_means_clusters)

	#load previously-made dictionary
	dictionary = load_file('dictionary_'+str(training_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))

	#fit dictionary
	neigh = NearestNeighbors(n_neighbors=k_neighbors)
	neigh.fit(dictionary)

	#------------------------------------------------------------------------------------------------------------



	#TODO: Why doesn't this work with 0?
	#Test getting an encoding
	index_of_descriptor = 1
	get_encoding(index_of_descriptor, training_descriptor_array, neigh, dictionary)




	'''#Codebook optimization -------------------------------------------------------------------------------------

	batch_size = 5
	batch_of_descriptors = descriptor_array[:batch_size,:]
	#batch_of_descriptors=descriptor_array

	optimize_codebook(dictionary, batch_of_descriptors, neigh)

	#-----------------------------------------------------------------------------------------------------------'''




	#TODO: Implement SPM with encodings

	#TODO: Pool and normalize SPM results

	#TODO: Use SPM results in classifier

	#Using approximated LLC encodings (skipping SPM step) for classifier
	#		=> Don't need to do pooling or normalization

	

	# create and save encodings for data -------------------------------------------------------------------

	training_descriptors = load_file("hog_descriptors_training_"+str(training_samples_per_class)+'_'+str(max_total_images))
	testing_descriptors = load_file("hog_descriptors_testing_"+str(testing_samples_per_class)+'_'+str(max_total_images))

	#run only if haven't already generated descriptors 
	if(generate_encodings):

		training_encodings = get_encodings_array(training_descriptors, neigh, dictionary)
		testing_encodings = get_encodings_array(testing_descriptors, neigh, dictionary)

		save_encodings(training_encodings, "encodings_training", training_samples_per_class, max_total_images, number_of_k_means_clusters)
		save_encodings(testing_encodings, "encodings_testing", testing_samples_per_class, max_total_images, number_of_k_means_clusters)

	training_encodings = load_file("encodings_training_"+str(training_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))
	testing_encodings = load_file("encodings_testing_"+str(testing_samples_per_class)+'_'+str(max_total_images)+'_'+str(number_of_k_means_clusters))

	# ------------------------------------------------------------------------------------------------------



	# Train classifier and predict results -----------------------------------------------------------------

	#TODO: Is classification better doing or not doing tranpose AFTER encodings (training) are done?

	classifier = train_classifier(total_classes, training_samples_per_class, training_encodings, training_target_array)

	accuracy = get_accuracy(classifier, testing_encodings, testing_target_array, testing_classification_dict)
	display("Accuracy is: ", accuracy)

	# -------------------------------------------------------------------------------------------------------



if __name__ == '__main__':

	# Datasets

	image_directory = "Caltech-101/101_ObjectCategories"
	test_image = "test_image.jpg"

	# Parameters

	k_neighbors = 5
	total_classes = 102
	training_samples_per_class = 30 					 					
	testing_samples_per_class = 50 						
	max_total_images = 10000
	number_of_k_means_clusters = 1024					#base of codebook


	main()


	
	
