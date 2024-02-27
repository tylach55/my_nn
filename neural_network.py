import numpy as np
import layer
import data_loader




class Neural_network:
	#num_layers will always be 2 (input and output layer)+ the number of hidden layers
	#input layer size will be the number of pixels in each image (28*28)
	#output layer size will be the number of discrete outputs (10) (0,1,2,3,4,5,6,7,8,9)
	#hidden layer sizes will be given by input 
	def __init__(self, layer_sizes):
		# self.layers = build_layers(layer_sizes)
		self.input_layer = None

		for i in range (0, len(layer_sizes)):
			if(i == 0):
				self.input_layer = layer.Layer(layer_sizes[i])
			else:
				new_layer = layer.Layer(layer_sizes[i])
				self.input_layer.add_layer(new_layer)

	def feed_forward(self, image):

		return self.input_layer.guess(image)

	def save_network(self, name):
		filepath = ("saved_networks/" + str(name))
		file = open(filepath, "w")
		i = 1

		current_layer = self.input_layer
		while(current_layer != None):
			file.write("\n====================================\n")
			file.write("layer " + str(i) + "\n")
			file.write("====================================\n")
			file.write("\n------------------------------------\n")
			file.write("neuron intensity\n")
			file.write("------------------------------------\n")
			file.write(str(current_layer.neuron_intensity))
			file.write("\n------------------------------------\n")
			file.write("connection weight\n")
			file.write("------------------------------------\n")
			file.write(str(current_layer.connection_weight))
			file.write("\n------------------------------------\n")
			file.write("bias\n")
			file.write("------------------------------------\n")
			file.write(str(current_layer.bias))
			i+=1
			current_layer = current_layer.next_layer


		file.close()




def main():
	print("extracting data from data/mnist.pkl.gz")
	training_data, validation_data, test_data = data_loader.load_data()
	training_data = list(training_data)
	input_layer_size = len(training_data[0][0])
	discrete_outputs = 10
	layer_sizes = [input_layer_size, 16, 16, discrete_outputs]
	for i in range (0,len(layer_sizes)):
		print("Layer " + str(i+1) + " is size " + str(layer_sizes[i]))
	print("building network")
	my_nn = Neural_network(layer_sizes)
	print(my_nn.feed_forward(training_data[0][0]))
	print("real number is " + str(training_data[0][1]))
	my_nn.save_network("network0")
main()