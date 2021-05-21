
from .neural import NeuralNetwork

class NeuralNetworkMomentum(NeuralNetwork):
    def __init__(self, input_count=1, layers_options=[], epochs=1000, learning_rate=2, epsilon=0.05, momentum=0.9):
        super().__init__(input_count, layers_options=layers_options, epochs=epochs, learning_rate=learning_rate, epsilon=epsilon)

        self.momentum = momentum
        self.previous_deltas = []

        for layer in self.layers:
            layer_previous_deltas = []
            
            for neuron in layer:
                init_weights_deltas = [0 for _ in neuron.weights]
                neuron_deltas = {
                    "weights": init_weights_deltas,
                    "bias": 0
                }
                layer_previous_deltas.append(neuron_deltas)

            self.previous_deltas.append(layer_previous_deltas)

    def update_weights(self):
        for layer_index, layer in enumerate(self.layers):
            for neuron_index, neuron in enumerate(layer):
                db = 2 * self.learning_rate * neuron.delta
                prev_db = self.previous_deltas[layer_index][neuron_index]["bias"]
                neuron.bias += (1 - self.momentum) * db + self.momentum * prev_db
                self.previous_deltas[layer_index][neuron_index]["bias"] = db

                for weight_index, weight in enumerate(neuron.weights):
                    dw = 2 * self.learning_rate * neuron.delta * neuron.input[weight_index]
                    prev_dw = self.previous_deltas[layer_index][neuron_index]["weights"][weight_index]
                    neuron.weights[weight_index] += (1 - self.momentum) * dw + self.momentum * prev_dw
                    self.previous_deltas[layer_index][neuron_index]["weights"][weight_index] = dw
        
    