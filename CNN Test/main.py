import numpy as np
import PIL as pil

from helpers import *
from layers import CNN, FC
import datasets as data
from function_layers import *

# Hyper params ----------------------------------------------------------------#
conv_hparams = {
"channels" : 1,
"f" : 5,
"stride" : 3,
"pad" : 1
}

conv_hparams2 = {
"channels" : 1,
"f" : 3,
"stride" : 3,
"pad" : 0
}

pool_hparams = {
"stride" : 1,
"pad" : 1,
"f" : 2
}

fully_con_hparams = {
"nodes" : 1
}


# Program ---------------------------------------------------------------------#

if __name__ == "__main__":
    train_X, train_Y = data.load_cats_dogs_64()
    train_X, train_Y = np.stack(train_X), np.reshape(train_Y, (-1, 1))
    train_X = train_X / 255

    cnn = CNN(conv_hparams)
    fc = FC(fully_con_hparams)

    Z1 = cnn.forward(train_X)
    A1 = relu(Z1)
    P1, cache = pool_forward(A1, pool_hparams)
    flatten = flatten(P1)
    Z2 = fc.forward(flatten)
    Y_hat = sigmoid(Z2)
    J = loss(Y_hat, train_Y)
    dZ4 = Y_hat - train_Y
    dZ2 = fc.backward(dZ4, Z2, "relu")
