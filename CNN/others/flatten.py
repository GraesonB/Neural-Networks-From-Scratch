import numpy as np

class Flatten:
    def __init__(self):
        self.shape = ()
        self.backward_out = None

    def forward(self, A):
        self.shape = A.shape
        data = np.ravel(A).reshape(self.shape[0], -1)
        return data

    def backward(self, dZ):
        self.backward_out = dZ.reshape(self.shape)
        return self.backward_out
