import numpy as np
import math

def sigmoid(npaarr):
	for i in range(0,len(npaarr)):
		for j in range(0, len(npaarr[i])):
			npaarr[i][j] = 1/(1+pow(math.e, -npaarr[i][j]))
	return npaarr
	

class Layer:
	"""
	-num_neurons is the number of neurons in this layer
	-num_connections is the number of neurons in the 
	 next layer that each neuron neeeds to make a connection with
	-neuron_intensity is an array of the values that each neuron is lit up at 
	-connection_weight is the weight for each connection each neuron makes
	-bias is the +- value to add to the function before the sigmoid
	-next_layer is the next layer

	"""
	def __init__(self, num_neurons):
		self.neuron_intensity = np.zeros((num_neurons, 1))
		self.connection_weight = None
		self.bias = None
		self.prev_layer = None
		self.next_layer = None

	def size(self):
		return len(self.neuron_intensity)

	def add_layer(self, new_layer):
		rng = np.random.default_rng()
		if self.next_layer == None:
			self.next_layer = new_layer
			new_layer.prev_layer = self
			self.next_layer = new_layer
			self.connection_weight = rng.standard_normal((new_layer.size(), self.size()))
			self.bias = rng.standard_normal((new_layer.size(),1))
		else:
			current_layer = self.next_layer
			while current_layer.next_layer is not None:
				current_layer = current_layer.next_layer
			new_layer.prev_layer = current_layer
			current_layer.next_layer = new_layer
			current_layer.connection_weight = rng.standard_normal((new_layer.size(), current_layer.size()))
			current_layer.bias = rng.standard_normal((new_layer.size(),1))
	
	def guess(self, new_neurons):
		self.neuron_intensity = new_neurons
		if self.next_layer == None:
			return new_neurons
		else:
			next_neurons = np.dot(self.connection_weight, self.neuron_intensity) + self.bias
			return self.next_layer.guess(sigmoid(next_neurons))



