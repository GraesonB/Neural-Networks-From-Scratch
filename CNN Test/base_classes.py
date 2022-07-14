import numpy as np
class Activation:
    def __init__(self):
        self.cache = None
        self. dx = None

class Layer:
    def __init(self):
        self.cache = None
        self.W = None
        self.b = None
        self.Z = None
        self.dW = None
        self.db = None
        self.dx = None
        # S matrices are for RMSProp, V matrices are for gradient descent with momentum, combine for Adam.
        self.SdW = None
        self.Sdb = None
        self.VdW = None
        self.Vdb = None
        self.time = 0

    def update_weights(self, learning_rate, beta_1, beta_2):
        epsilon = 0.000001
        self.time += 1
        # V terms are for GD with momentum, S terms are for RMS prop, as well as beta 1 and 2 respectively
        self.VdW = beta_1 * self.VdW + (1 - beta_1) * self.dW
        self.Vdb = beta_1 * self.Vdb + (1 - beta_1) * self.db
        self.SdW = beta_2 * self.SdW + (1 - beta_2) * self.dW ** 2
        self.Sdb = beta_2 * self.Sdb + (1 - beta_2) * self.db ** 2
        VdW_corrected = self.VdW / (1 - beta_1 ** self.time)
        Vdb_corrected = self.Vdb / (1 - beta_1 ** self.time)
        SdW_corrected = self.SdW / (1 - beta_2 ** self.time)
        Sdb_corrected = self.Sdb / (1 - beta_2 ** self.time)

        # self.W = self.W - learning_rate * self.dW
        # self.b = self.b - learning_rate * self.db
        self.W = self.W - (learning_rate * VdW_corrected / (np.sqrt(SdW_corrected) + epsilon))
        self.b = self.b - (learning_rate * Vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon))
