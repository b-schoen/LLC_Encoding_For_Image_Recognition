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