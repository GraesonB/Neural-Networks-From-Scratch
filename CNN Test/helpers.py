import numpy as np

def zero_pad(X, pad):
    # X - numpy array of shape (m, width, height, # of channels)
    # pad - is an integer that pads each image around the width and height with 0s
    X_pad = np.pad(X, ((0,0), (pad,pad), (pad,pad), (0,0)), mode='constant', constant_values = (0,0))
    return X_pad

def single_convolution(a_slice_prev, W, b):
    # Z - a scalar value, the result of convolving the sliding window (W, b) on a slice x of the input data
    s = np.multiply(a_slice_prev, W)
    Z = np.sum(s)
    Z = np.float64(Z + b)
    return Z

def create_mask_from_window(x):
    mask = (x == np.max(x))
    return mask

def flatten(X):
    return np.reshape(X, (np.shape(X)[0],-1))

def loss(Y_hat, Y):
    return -1/len(Y) * np.sum(Y * np.log(Y_hat) + (1-Y) * (np.log(1-Y_hat)))


def relu(Z):
    return Z * (Z > 0)

def sigmoid(Z):
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid(Z) * (1 - sigmoid(Z))

def relu_derivative(Z):
    return (Z >= 0);

if __name__ == "__main__":
    pass
