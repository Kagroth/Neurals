
import numpy as np
import random
import time
from termcolor import colored
from icecream import ic
from .activation_functions import sigmoid, deriv_sigmoid

def generate_weights(neuron_count=1, weights_per_neuron=1):
    return np.random.rand(neuron_count, weights_per_neuron)

class Neuron:
    def __init__(self, weights=None, input_count=0, bias=0, activation_function=sigmoid, deriv_function=deriv_sigmoid):
        self.input_count = input_count
        self.input = None
        self.bias = bias
        self.activation_function = activation_function
        self.deriv_function = deriv_function
        self.output = None
        self.delta = None

        if weights is None:
            self.weights = np.random.uniform(low=-1, high=1, size=input_count)
        else:
            self.weights = weights

        try:
            if self.input_count != self.weights.shape[0]:
                raise
        except:
            print(colored("Długość wektora wag musi być równa ilości wejść neuronu", "red"))
            print(colored("input_count: {}\n weights length: {}".format(self.input_count, self.weights.shape), "red"))
            return

    def feed_forward(self, input_data):
        s = np.dot(input_data, self.weights.T) + self.bias
        self.input = input_data
        self.output = self.activation_function(s)
        
        return self.output

    def __str__(self):
        return "{} {}, \n Wagi: {}, \n Bias: {}, \n Aktywacja: {} \n Wynik neuronu: {} \n Delta: {}".format(self.__class__.__name__, 
                                                                                                    hex(id(self)), 
                                                                                                    self.weights, 
                                                                                                    self.bias, 
                                                                                                    self.activation_function.__name__,
                                                                                                    self.output,
                                                                                                    self.delta) 

class LayerOption:
    def __init__(self, neuron_count=1, weights=None, activation_function=sigmoid, deriv_function=deriv_sigmoid):
        self.neuron_count = neuron_count
        self.weights = weights
        self.activation_function = activation_function
        self.deriv_function = deriv_function

class NeuralNetwork:
    def __init__(self, input_count=1, layers_options=[], epochs=1000, learning_rate=2, epsilon=0.05):
        self.input_count = input_count
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.epsilon = epsilon
        self.layers = []

        for index, layer_option in enumerate(layers_options):
            layer = []
            input_size = 0
        
            if index == 0:
                input_size = self.input_count
            else:
                input_size = layers_options[index - 1].neuron_count

            for i in range(layer_option.neuron_count):
                weights_to_set = None

                if layer_option.weights is not None:
                    weights_to_set = layer_option.weights[i]
                
                neuron = Neuron(input_count=input_size, 
                                weights=weights_to_set, 
                                activation_function=layer_option.activation_function, 
                                deriv_function=layer_option.deriv_function)
                layer.append(neuron)
            
            assert len(layer) == layer_option.neuron_count

            self.layers.append(layer)

        assert len(self.layers) == len(layers_options) 

    def feed_forward(self, input_data):
        layer_input = input_data
        
        for layer in self.layers:
            layer_output = []

            for neuron in layer:
                neuron.input = layer_input
                output = neuron.feed_forward(layer_input)
                layer_output.append(output)

            layer_input = np.array(layer_output)
        
        self.output = layer_output
        
        return self.output
    
    def update_weights(self):
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer):
                neuron.bias += 2 * self.learning_rate * neuron.delta

                for weight_index, weight in enumerate(neuron.weights):
                    dw = 2 * self.learning_rate * neuron.delta * neuron.input[weight_index]
                    neuron.weights[weight_index] += dw

    def backpropagation(self, input_data, output_data):
        # Propagacja błędu od warstwy wyjściowej do warstwy wejściowej
        # Zapis "reversed(list(enumerate(self.layers)))" pozwala na iterację 
        # od końca tj. od L do 0, gdzie L - liczba warstw
        # alternatywny zapis to "reversed(range(len(self.layers)))"
        for layer_index, layer in reversed(list(enumerate(self.layers))):
            if layer_index + 1 == len(self.layers):
                for neuron_index, neuron in enumerate(layer):
                    s = np.dot(neuron.input, neuron.weights.T) + neuron.bias
                    neuron.delta = (output_data[neuron_index] - neuron.output) * neuron.deriv_function(s)
            else:
                for neuron_index, neuron in enumerate(layer):
                    s = np.dot(neuron.input, neuron.weights.T) + neuron.bias
                    next_layer = self.layers[layer_index + 1]
                    neuron.delta = 0
                    for nl_neuron in next_layer:
                        neuron.delta += nl_neuron.delta * nl_neuron.weights[neuron_index]
                    neuron.delta *= neuron.deriv_function(s)

    def train(self, input_data, true_output):
        error_history = []

        for epoch_index in range(self.epochs):
            outputs = []
            global_error = 0

            for input_vector, output_vector in zip(input_data, true_output):
                self.feed_forward(input_vector)
                self.backpropagation(input_vector, output_vector)
                self.update_weights()

            outputs = np.apply_along_axis(self.feed_forward, 1, input_data)
            global_error = self.mse_loss(true_output, outputs)
            
            error_history.append(global_error)

            if global_error < self.epsilon:
                return error_history
            
        return error_history

    def mse_loss(self, true_output, current_output):
        loss = ((true_output - current_output) ** 2).mean()
        return loss

    def __str__(self):
        t = "Sieć neuronowa {} \n".format(hex(id(self)))

        for index, layer in enumerate(self.layers):
            t = t + "Warstwa nr {} \n".format(index) 
            
            for neuron in layer:
                t = t + neuron.__str__() + "\n"
            
        return t