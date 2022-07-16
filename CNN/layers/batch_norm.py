from others.helpers import *
from others.base_classes import Layer

# inherits from Layer only because it has weights for the weight update check
class BatchNorm(Layer):
    def __init__(self):
        self.time = 0
        self.has_weights = True
        self.gamma = None
        self.beta = None
        self.cache = None
        self.z_norm = None
        self.forward_out = None
        self.gamma_grads = None
        self.beta_grads = None
        self.Vd_gamma = None
        self.Vd_beta = None
        self.Sd_gamma = None
        self.Sd_beta = None

    def forward(self, Z, epsilon = 1e-8):
        self.cache = Z
        self.mean = np.mean(Z, axis = 0)
        self.variance = np.var(Z, axis = 0)
        self.z_norm = (Z - self.mean)/np.sqrt(self.variance + epsilon)
        self.forward_out = self.gamma * self.z_norm + self.beta
        return self.forward_out

    def backward(self, dA, epsilon = 1e-8):
        m = dA.shape[0]
        t = 1 / np.sqrt(self.variance + epsilon)
        self.beta_grads = np.sum(dA, axis = 0)
        self.gamma_grads = np.sum((self.z_norm * dA), axis = 0)
        self.backward_out = (self.gamma * t / m) * (m * dA - np.sum(dA, axis=0)
         - t**2 * (self.cache-self.mean) * np.sum(dA*(self.cache - self.mean), axis=0))
        return self.backward_out

    def initialize_matrices(self, A_prev):
        self.gamma = np.ones(A_prev.shape[1:])
        self.beta = np.zeros(A_prev.shape[1:])
        self.gamma_grads = np.zeros_like(self.gamma)
        self.beta_grads = np.zeros_like(self.gamma)
        self.Vd_gamma = np.zeros_like(self.gamma)
        self.Vd_beta = np.zeros_like(self.gamma)
        self.Sd_gamma = np.zeros_like(self.gamma)
        self.Sd_beta = np.zeros_like(self.gamma)
        self.forward_out = np.zeros_like(A_prev)
        return self.forward_out

    def update_weights(self, learning_rate, beta_1, beta_2, epsilon = 1e-8):
        self.time += 1
        # V terms are for GD with momentum, S terms are for RMS prop, as well as beta 1 and 2 respectively
        self.Vd_gamma = beta_1 * self.Vd_gamma + (1 - beta_1) * self.gamma_grads
        self.Vd_beta = beta_1 * self.Vd_beta + (1 - beta_1) * self.beta_grads
        self.Sd_gamma = beta_2 * self.Sd_gamma + (1 - beta_2) * self.gamma_grads ** 2
        self.Sd_beta = beta_2 * self.Sd_beta + (1 - beta_2) * self.beta_grads ** 2
        Vd_gamma_corrected = self.Vd_gamma / (1 - beta_1 ** self.time)
        Vd_beta_corrected = self.Vd_beta / (1 - beta_1 ** self.time)
        Sd_gamma_corrected = self.Sd_gamma / (1 - beta_2 ** self.time)
        Sd_beta_corrected = self.Sd_beta / (1 - beta_2 ** self.time)

        # self.weights = self.weights - learning_rate * self.weight_grads
        # self.biases = self.biases - learning_rate * self.bias_grads
        self.gamma = self.gamma - (learning_rate * Vd_gamma_corrected / (np.sqrt(Sd_gamma_corrected) + epsilon))
        self.beta = self.beta - (learning_rate * Vd_beta_corrected / (np.sqrt(Sd_beta_corrected) + epsilon))
