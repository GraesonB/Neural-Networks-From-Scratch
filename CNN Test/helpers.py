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
    return np.ravel(X).reshape(X.shape[0], -1)

def he_uniform(shape):
    '''
    A function for smart uniform distribution based initialization of parameters
    [He et al. https://arxiv.org/abs/1502.01852]
    :param fan_in: The number of units in previous layer.
    :param fan_out: The number of units in current layer.
    :return:[numpy array, numpy array]: A randomly initialized array of shape [fan_out, fan_in] and
            the bias of shape [fan_out, 1]
    '''
    scale = np.sqrt(6. / shape[0])
    bias_shape = (shape[1], 1)
    weights = np.random.uniform(-scale, scale, size=shape)
    return weights

def binary_cross_entropy_loss(Y_pred, Y):
    m = Y_pred.shape[0]
    clipped_pred = np.clip(Y_pred, a_min = 1e-8, a_max = None)
    loss = -1/(Y.shape[0]) * np.sum(Y * np.log(clipped_pred) + (1-Y) * (np.log(1-clipped_pred)))
    accuracy = 1 - (np.sum(np.square(np.round(Y_pred) - Y)) / m)

    return loss, accuracy

def softmax_loss(Y_pred, Y, epsilon = 0.0000001):
    Y_pred /= np.sum(Y_pred, axis=0, keepdims=True)
    Y_pred = np.clip(Y_pred, epsilon, 1. - epsilon)
    return -np.sum(Y * np.log(Y_pred))

def get_one_hot(targets, nb_classes):
    res = np.eye(nb_classes)[np.array(targets).reshape(-1)]
    return res.reshape(list(targets.shape)+[nb_classes])

def relu(Z):
    return Z * (Z > 0)

def sigmoid_f(Z):
    #Z = Z.astype(float)
    return 1 / (1 + np.exp(-Z))

def sigmoid_derivative(Z):
    return sigmoid_f(Z) * (1 - sigmoid_f(Z))

def relu_derivative(Z):
    return (Z >= 0)

if __name__ == "__main__":
    pass
