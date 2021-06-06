
import numpy as np

class KohonenNeuron:
    def __init__(self, input_size=1, weights=None, bias=0):
        self.weights = weights
        self.bias = bias
        self.output = None

        if weights is None:
            # self.weights = np.random.uniform(low=-1, high=1, size=input_size)
            self.weights = np.array([0.0, 0.0])
    
    def distance(self, point):
        subs = np.subtract(point, self.weights)
        pow = np.power(subs, 2)
        s = np.sum(pow)
        output = np.sqrt(s)
        self.output = output
        return output


class KohonenNet:
    def __init__(self, input_size=1, layer_size=(1, 1), epochs=100, learning_rate=0.2):
        self.epochs = epochs
        self.initial_learning_rate = learning_rate
        self.learning_rate = learning_rate
        self.layer = [] # macierz neuronow o rozmiarze layer_size

        for x in range(layer_size[0]):
            self.layer.append([])
            for _ in range(layer_size[1]):
                kn = KohonenNeuron(input_size=input_size)
                self.layer[x].append(kn)
    
    def update_learning_rate(self, current_epoch):
        self.learning_rate = self.initial_learning_rate / self.epochs * (self.epochs - current_epoch)

    def update_weights(self, point, pos):
        x, y = pos
        # zmiana zwycieskiego neuronu - tego ktory jest najblizej wektora wejsciowego (point)
        dw = self.learning_rate * (point - self.layer[x][y].weights)
        self.layer[x][y].weights += dw

        # zmiana sasiednich neuronow, polaczenie miedzy neuronami typu grid
        # wspolrzedne sasiadujacych neuronow:
        # x, y+1
        if y < len(self.layer[x]) - 1:
            dw = self.learning_rate * (point - self.layer[x][y+1].weights)
            self.layer[x][y+1].weights += dw
        
        # x+1, y
        if x < len(self.layer) - 1:
            dw = self.learning_rate * (point - self.layer[x+1][y].weights)
            self.layer[x+1][y].weights += dw

        # x-1, y
        if x > 0:
            dw = self.learning_rate * (point - self.layer[x-1][y].weights)
            self.layer[x-1][y].weights += dw

        # x, y-1
        if y > 0:
            dw = self.learning_rate * (point - self.layer[x][y-1].weights)
            self.layer[x][y-1].weights += dw


    def train(self, input_data):
        best_neuron_pos = (0, 0)
        best_distance = float("inf")
        for epoch in range(self.epochs):
            self.update_learning_rate(epoch)
            for input_vector in input_data:
                best_distance = float("inf")
                for x in range(len(self.layer)):
                    for y in range(len(self.layer[x])):
                        d = self.layer[x][y].distance(input_vector)
                        if d < best_distance:
                            best_distance = d
                            best_neuron_pos = (x, y)
                
                # update weights
                self.update_weights(input_vector, best_neuron_pos)