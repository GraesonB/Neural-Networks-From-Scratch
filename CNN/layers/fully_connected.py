from others.helpers import *
from others.base_classes import Layer

class FCLayer(Layer):
    def __init__(self, hparameters):
        super().__init__()
        self.nodes = hparameters["nodes"]

        self.has_weights = True
        self.time = 0

    def forward(self, A_previous):
        self.cache = A_previous
        self.forward_out = np.dot(A_previous, self.weights) + self.biases
        return self.forward_out

    def backward(self, dout):
        m = self.cache.shape[0]
        self.weight_grads = (1/m) * np.dot(self.cache.T, dout)
        # keep dims for bias_grads to avoid creating rank-1 array
        self.bias_grads = (1/m) * np.sum(dout, axis = 0, keepdims = True)
        self.backward_out = np.dot(dout, self.weights.T)
        return self.backward_out

    def initialize_matrices(self, A_previous):
        if A_previous.ndim != 2:
            A_previous = flatten(A_previous)
        n = A_previous.shape[1] * self.nodes + 1
        self.weights = he_uniform((A_previous.shape[1], self.nodes))
        self.biases = np.zeros((1,self.nodes))
        self.SdW = np.zeros_like(self.weights)
        self.Sdb = np.zeros_like(self.biases)
        self.VdW = np.zeros_like(self.weights)
        self.Vdb = np.zeros_like(self.biases)
        self.forward_out = np.zeros((A_previous.shape[0], self.weights.shape[1]))
        return self.forward_out
