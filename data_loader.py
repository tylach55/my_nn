import numpy as np
import pickle
import gzip

def load_data():

	"""
	!!!!stolen code!!!!
	opens the data set that I got from mneilsen on github
	returns a tuple of training data, validation data, and test data
	each datais a tuple of 
		a 28*28 ndaaray
		a label
	the 28*28 ndaaray is turned into a 784*1 ndarray

	training_data has 50,000 entries and test and validation have 10,000
	I am also reordering the inputs from the data set from a 28*28 to 784*1
	
	mneilsen also "vectorizes" the labels on his code, that is to make back
	propogation easier
	"""
	f = gzip.open('data/mnist.pkl.gz', 'rb')
	tr_d, va_d, te_d = pickle.load(f, encoding="latin1")
	f.close()

	training_inputs = [np.reshape(x, (784, 1)) for x in tr_d[0]]
	training_data = zip(training_inputs, tr_d[1])

	validation_inputs = [np.reshape(x, (784, 1)) for x in va_d[0]]
	validation_data = zip(validation_inputs, tr_d[1])

	test_inputs = [np.reshape(x, (784, 1)) for x in te_d[0]]
	test_data = zip(test_inputs, tr_d[1])

	return (training_data, validation_data, test_data)

