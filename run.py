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
from sklearn.svm import SVC

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
	

def find_local_basis(desc, neigh, dictionary):
	ind=neigh.kneighbors(desc)[1]
	B=dictionary[ind,:]
	return np.squeeze(B)



#def show_hog():
#
#	fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(8, 4), sharex=True, sharey=True)
#
#	ax1.axis('off')
#	ax1.imshow(image, cmap=plt.cm.gray)
#	ax1.set_title('Input image')
#	ax1.set_adjustable('box-forced')
#
#	# Rescale histogram for better display
#	hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 0.02))
#
#	ax2.axis('off')
#	ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
#	ax2.set_title('Histogram of Oriented Gradients')
#	ax1.set_adjustable('box-forced')
#	plt.show()

def get_hog_descriptor(image):

	orientations = 9
	pixels_per_cell = (8,8)
	cells_per_block = (2,2)
	return_as_feature_vector = True

	feature_descriptor, hog_image = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block, visualise=True)

	return hog_image

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

def run_on_all_images(process_image, image_directory, images_to_sample_per_class, number_to_stop_at):

	#TODO: Make sure these are being run on images in the correct order

	#specify generic form of return data
	return_data = []

	#number of images processed
	images_processed = 0
	#number of images to sample per class
	images_sampled_in_class = 0

	print(image_directory)

	for filename in os.listdir(image_directory):
	#with open(image_directory) as current_file:

		print(filename)

		filename = image_directory +"/" + filename

		images_sampled_in_class = 0

		for image_file in os.listdir(filename):

			if(images_sampled_in_class < images_to_sample_per_class):

				#print(image_file)

				image_file = filename +"/" + image_file

				image = Image.open(image_file)

				image = prepare_image(image)

				flattened_descriptor = process_image(image)

				if(images_processed==0):
					return_data=(flattened_descriptor)
				else:
					return_data = np.vstack((return_data,flattened_descriptor))
				#describe_array(return_data)
				describe_array(return_data)

				#describe_array(flattened_descriptor)

				#return if reached max images total to process
				images_processed = images_processed + 1
				print ("Processed ",str(images_processed)," images so far")
				if(images_processed >= number_to_stop_at):
					return return_data

				line_break()
				line_break()

			images_sampled_in_class = images_sampled_in_class+1

			
	
	return return_data

def pooling(list_of_vectors, type_of_method):

	#sum of vectors
	if(type_of_method=="sum"):

		#NOTE: Copy because otherwise numpy goes by reference
		total_sum = np.copy(list_of_vectors[0])

		for index in range(1, len(list_of_vectors)):

			total_sum = total_sum + list_of_vectors[index]

		return total_sum

	#vector of row maxes
	elif(type_of_method=="max"):

		#initially set max_vector to first vector, so know size to iterate over
		#NOTE: Copy because otherwise numpy goes by reference
		max_vector = np.copy(list_of_vectors[0])

		#for each row
		for row_value_index in range(max_vector.size):

			row_value_list = []

			#get the row value of each vector
			for list_index in range(0,len(list_of_vectors)):

				current_row_value = (list_of_vectors[list_index])[row_value_index]

				row_value_list.append(current_row_value)

			#set the row value of max_vector to be the biggest row value out of all row values for vectors
			max_vector[row_value_index] = max(row_value_list)

		return max_vector

	else:
		raise NameError("pooling type_of_method must be either sum or max")


def normalize(input_vector, type_of_method):

	if(type_of_method=="sum"):

		sum_of_vector_elements = np.sum(input_vector)
		
		normalized_vector = input_vector * (1.0 / sum_of_vector_elements)

		return normalized_vector

	elif(type_of_method=="l_2"):
		
		#vector norm defaults to 2-norm
		l_2_norm = np.linalg.norm(input_vector)
		
		normalized_vector = input_vector * (1.0 / l_2_norm)

		return normalized_vector

	else:
		raise NameError("normalize type_of_method must be either sum or l_2")

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

def train_SVM():

	pass

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

def main():

	# Datasets

	image_directory = "Caltech-101/101_ObjectCategories"

	k_neighbors = 5
	samples_per_class = 50 #needs to be changed to not overlap with training set (ex: first 5 for training means can't just take first 50 for testing)
	max_total_images = 100000
	number_of_k_means_clusters = 1024		#base of codebook

	#run only if haven't already generated descriptors 
	#create and save new array of descriptors -------------------------------------------------------

	#get descriptors
	#descriptor_list = run_on_all_images(get_hog_descriptor, image_directory, samples_per_class, max_total_images)

	#convert to array
	#descriptor_array = np.array(descriptor_list)

	#save array so don't have to recalcualte each time
	#save_descriptor_array(descriptor_array, 'hog_descriptors_testing', samples_per_class, max_total_images)

	#------------------------------------------------------------------------------------------------------------


 	#load previously-made array of descriptors
	descriptor_array = load_file('hog_descriptors_5_100000')

	#run only if haven't already generated descriptors 

	#create & save new dictionary -------------------------------------------------------------------------------

	#dictionary = do_k_means((descriptor_array), number_of_k_means_clusters)

	#save_dictionary(dictionary, 'dictionary', samples_per_class, max_total_images, number_of_k_means_clusters)

	#------------------------------------------------------------------------------------------------------------

	#load previously-made dictionary
	dictionary = load_file('dictionary_5_100000_1024')

	neigh = NearestNeighbors(n_neighbors=k_neighbors)
	neigh.fit(dictionary)

	#TODO: Why doesn't this work with 0?
	index_of_descriptor = 1
	get_encoding(index_of_descriptor, descriptor_array, neigh, dictionary)

	batch_size = 5
	batch_of_descriptors = descriptor_array[:batch_size,:]
	#batch_of_descriptors=descriptor_array

	optimize_codebook(dictionary, batch_of_descriptors, neigh)


if __name__ == '__main__':

	main()
	
	
