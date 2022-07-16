import numpy as np
class Activation:
    def __init__(self):
        self.cache = None
        self. dx = None

class Layer:
    def __init(self):
        self.cache = None
        self.weights = None
        self.biases = None
        self.weight_grads = None
        self.bias_grads= None

        self.forward_out = None
        self.backward_out = None

        # S matrices are for RMSProp, V matrices are for gradient descent with momentum, combine for Adam.
        self.SdW = None
        self.Sdb = None
        self.VdW = None
        self.Vdb = None
        self.time = 0

    def update_weights(self, learning_rate, beta_1, beta_2, epsilon = 1e-8, optimizer = "adam"):
        self.time += 1
        if optimizer == "adam":
            # V terms are for GD with momentum, S terms are for RMS prop, as well as beta 1 and 2 respectively
            self.VdW = beta_1 * self.VdW + (1 - beta_1) * self.weight_grads
            self.Vdb = beta_1 * self.Vdb + (1 - beta_1) * self.bias_grads
            self.SdW = beta_2 * self.SdW + (1 - beta_2) * self.weight_grads ** 2
            self.Sdb = beta_2 * self.Sdb + (1 - beta_2) * self.bias_grads ** 2
            VdW_corrected = self.VdW / (1 - beta_1 ** self.time)
            Vdb_corrected = self.Vdb / (1 - beta_1 ** self.time)
            SdW_corrected = self.SdW / (1 - beta_2 ** self.time)
            Sdb_corrected = self.Sdb / (1 - beta_2 ** self.time)

            self.weights = self.weights - (learning_rate * VdW_corrected / (np.sqrt(SdW_corrected) + epsilon))
            self.biases = self.biases - (learning_rate * Vdb_corrected / (np.sqrt(Sdb_corrected) + epsilon))
        else:
            self.weights = self.weights - learning_rate * self.weight_grads
            self.biases = self.biases - learning_rate * self.bias_grads
