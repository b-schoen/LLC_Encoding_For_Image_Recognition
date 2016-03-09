## Steps
#	Get HOG descriptor
#	Generate codebook via k-means
#	Use LLC to futher optimize codebook (pseudocode provided)

#matlab plotting
#import matplotlib.pyplot as plt

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
from utilities import describe_array
from utilities import line_break
from utilities import save_dictionary
from utilities import save_descriptor_array
from utilities import save_encodings
from utilities import load_file
from utilities import prepare_image
from utilities import flatten_daisy

# functions from SPM.py
from SPM import pooling
from SPM import normalize

def do_k_means(input_data, clusters):

	print("Doing k means")

	#scale data
	scaled_data = scale(input_data)
	describe_array(scaled_data)

	#get samples and features
	n_samples, n_features = scaled_data.shape

	return bench_k_means(KMeans(init='k-means++', n_clusters=clusters, n_init=10, verbose=1),
              name="k-means++", data=scaled_data)

def bench_k_means(estimator, name, data):
    t0 = time()
    km=estimator.fit(data)

    print("Done k means")

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

def get_hog_descriptor(image):

	orientations = 9
	pixels_per_cell = (8,8)
	cells_per_block = (2,2)
	return_as_feature_vector = True

	feature_descriptor, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=True)

	return hog_image

def run_on_all_images(process_image, image_directory, images_to_sample_per_class, number_to_stop_at, calc_descriptors):

	#TODO: Make sure these are being run on images in the correct order

	#specify generic form of return data
	return_data = []

	classification_list = []
	classification_dict = {}
	class_count = 0

	#number of images processed
	images_processed = 0
	#number of images to sample per class
	images_sampled_in_class = 0

	print(image_directory)

	for filename in os.listdir(image_directory):
	#with open(image_directory) as current_file:

		print(filename)

		current_class = filename

		classification_dict[current_class] = class_count

		filename = image_directory +"/" + filename

		images_sampled_in_class = 0

		for image_file in os.listdir(filename):

			if(images_sampled_in_class < images_to_sample_per_class):

				#print(image_file)

				image_file = filename +"/" + image_file

				image = Image.open(image_file)

				image = prepare_image(image)

				if(calc_descriptors):

					descriptor = process_image(image)

					#print("Resulting descriptor is: ")
					#describe_array(descriptor)

					if(images_processed==0):
						return_data=descriptor

					else:
						return_data = np.vstack((return_data, descriptor))

					describe_array(return_data)

					#return if reached max images total to process
					images_processed = images_processed + 1
					print ("Processed ",str(images_processed)," images so far")
					if(images_processed >= number_to_stop_at):
						return return_data

				#log classification of this images
				classification_list.append(class_count)

				line_break()
				line_break()

			images_sampled_in_class = images_sampled_in_class+1

		#done with this class
		class_count = class_count + 1

	print(classification_list)
	print(classification_dict)

	#convert to arrays
	target_array = np.asarray(classification_list)
	return_array = np.array(return_data)

	describe_array(return_array)

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

	print("Codebook has shape: ", codebook.shape)

	print("Batch of descriptors has shape: ", batch_of_descriptors.shape)

	# D - dimension of local descriptors
	D = batch_of_descriptors.shape[1]

	# N - number of descriptors in this batch of the online learning algorithm
	#		-	This is why X is DxN, because it's an array of N descriptor vectors
	N = batch_of_descriptors.shape[0]

	# M - number of entries (vectors) in codebook
	M = codebook.shape[0]

	print("D is: ", D)
	print("N is: ", N)
	print("M is: ", M)

	# --------------------------------------------------------------------------------------

 	##B ← B init .
 	optimized_codebook = np.copy(codebook)

 	##for i = 1 to N do
 	for i in range(1,N):

 		print("-------------------------------------------------------------------------------------")
 		print(i)

 		x_i = batch_of_descriptors[i,:]
 		print(x_i.shape)

		##	d ← 1 × M zero vector, 
		d = np.zeros((1,M))

		##{locality constraint parameter} ------------------

		print("{locality constraint parameter} ------------------")

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

		print("{coding} -----------------------------------------")

		##c i ← argmax c ||x i − Bc|| 2 + λ||d c|| 2 s.t. 1 c = 1. 
		#TODO: Assuming coding here can just use K-NN to approx like we did earlier
		c_i = get_encoding(i, batch_of_descriptors, neigh, codebook)
		print(c_i)

		##{remove bias} ------------------------------------

		print("{remove bias} ------------------------------------")

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
		print(B_i.shape)

		##c i ← argmax c ||x i − B i c|| 2 s.t. j c(j) = 1.
		#TODO: This is without that constraint
		tilde_c_i = np.linalg.lstsq(np.transpose(B_i),np.transpose(x_i))[0]

		##{update basis} -----------------------------------

		print("{update basis} -----------------------------------")

		##ΔB i ←= −2 c i (x i − B i  ̃ c i )
		#TODO: How do I enforce that constraint?
		#delta_B_i =  -2*tilde_c_i*(x_i-B_i*tilde_c_i)

		##μ ← 1/i
		#mu = 1.0 / i

		##B i ← B i − μΔB i /| ̃c i | 2

		##B(:, id) ← proj(B i ).


	##output: B

	return "in progress"


def label_images():

	print("in progress")

def find_local_basis(descriptor, neigh, dictionary):
	ind=neigh.kneighbors(descriptor)[1]
	B=dictionary[ind,:]
	return np.squeeze(B)

def get_encoding(index_of_descriptor, descriptor_array, neigh, dictionary):

	print("Getting encoding")

	#testing encoding on 1 descriptor at index_of_descriptor
	xi = descriptor_array[index_of_descriptor].reshape(1,-1)
	bi = find_local_basis(xi, neigh, dictionary)
	ci = np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[0] #k-length encoding
	err= np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[1]

	#display results
	'''print("Descriptor for x_i: ")
	describe_array(xi)
	print(xi)

	print("Base for x_i: ")
	describe_array(bi)
	print(bi)

	print("Code for x_i: ")
	describe_array(ci)
	print(ci)

	print("Error for x_i: ")
	describe_array(err)
	print(err)'''

	return ci

def get_descriptor_encoding(descriptor, neigh, dictionary):

	xi = descriptor.reshape(1,-1)
	bi = find_local_basis(xi, neigh, dictionary)
	ci = np.linalg.lstsq(np.transpose(bi),np.transpose(xi))[0] #k-length encoding

	return ci

def train_classifier(total_classes, samples_per_class, training_data, target):

	print("Target looks like: ")

	describe_array(target)

	target.reshape(total_classes*samples_per_class,1)

	linear_svc = LinearSVC()
	
	linear_svc.fit(training_data, target)

	return linear_svc

def get_encodings_array(descriptor_array, neigh, dictionary):

	encoding_list = []

	#TODO: Is it rows instead?
	for column in descriptor_array:

		encoding_list.append(get_descriptor_encoding(column, neigh, dictionary))

	return np.squeeze(np.asarray(encoding_list))

def main():

	# Datasets

	image_directory = "Caltech-101/101_ObjectCategories"

	k_neighbors = 5
	total_classes = 102
	samples_per_class = 5 					#needs to be changed to not overlap with training set (ex: first 5 for training means can't just take first 50 for testing)
	max_total_images = 10000
	number_of_k_means_clusters = 1024		#base of codebook
	calc_descriptors = False

	#run only if haven't already generated descriptors 
	#create and save new array of descriptors -------------------------------------------------------

	#get descriptors
	descriptor_array, target_array, classification_dict = run_on_all_images(get_hog_descriptor, image_directory, samples_per_class, max_total_images, calc_descriptors)

	#save array so don't have to recalcualte each time
	if(calc_descriptors):
		save_descriptor_array(descriptor_array, 'hog_descriptors', samples_per_class, max_total_images)

	#------------------------------------------------------------------------------------------------------------

 	#load previously-made array of descriptors
	descriptor_array = load_file('hog_descriptors_5_10000')

	#run only if haven't already generated descriptors 

	'''#create & save new dictionary -------------------------------------------------------------------------------

	dictionary = do_k_means((descriptor_array), number_of_k_means_clusters)

	save_dictionary(dictionary, 'dictionary', samples_per_class, max_total_images, number_of_k_means_clusters)

	#------------------------------------------------------------------------------------------------------------'''

	#load previously-made dictionary
	dictionary = load_file('dictionary_5_100000_1024')

	neigh = NearestNeighbors(n_neighbors=k_neighbors)
	neigh.fit(dictionary)

	#TODO: Why doesn't this work with 0?
	index_of_descriptor = 1
	get_encoding(index_of_descriptor, descriptor_array, neigh, dictionary)

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

	#run only if haven't already generated descriptors 

	# create and save encodings -----------------------------------------------------------------------------

	#training_descriptors = load_file("hog_descriptors_5_10000")

	#encodings = get_encodings_array(training_descriptors, neigh, dictionary)

	#save_encodings(encodings, "encodings", samples_per_class, max_total_images, number_of_k_means_clusters)

	# ------------------------------------------------------------------------------------------------------

	encodings = load_file("encodings_5_10000_1024")

	describe_array(encodings)

	#TODO: Need correct target vector for this
	classifier = train_classifier(total_classes, samples_per_class, encodings, target_array)

	#TODO: Copy hog source code into some other file then modify it



if __name__ == '__main__':

	main()
	
	
