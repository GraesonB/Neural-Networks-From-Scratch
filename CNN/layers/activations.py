from others.helpers import *
from others.base_classes import Activation

class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        return np.where(Z >= 0, Z, 0)

    def backward(self, dA):
        self.backward_out = (self.cache >= 0)
        self.backward_out = dA * np.where(self.cache >= 0, 1, 0)
        return self.backward_out

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        self.output = sigmoid_f(Z)
        return self.output

    def backward(self, dA):
        self.backward_out = np.multiply(np.multiply(self.output, (1 - self.output)), dA)
        return self.backward_out
