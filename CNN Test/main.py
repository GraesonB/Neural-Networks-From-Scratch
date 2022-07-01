import numpy as np
import PIL as pil

from base_classes import *
from modules import *
import datasets as data
from function_layers import *

# Hyper params ----------------------------------------------------------------#
conv_hparams = {
"channels" : 3,
"f" : 5,
"stride" : 2,
"pad" : 1
}

conv_hparams2 = {
"channels" : 4,
"f" : 5,
"stride" : 2,
"pad" : 0
}

pool_hparams = {
"stride" : 2,
"pad" : 1,
"f" : 2
}

fully_con_hparams = {
"nodes" : 20
}

fully_con_hparams2 = {
"nodes" : 1
}

learning_rate = 0.07

# Program ---------------------------------------------------------------------#

if __name__ == "__main__":
    train_X, train_Y = data.load_cats_dogs_300()
    train_X, train_Y = np.stack(train_X), np.reshape(train_Y, (-1, 1))
    train_X = train_X / 255
    cnn = CNN(conv_hparams)
    relu = Relu()
    pool = Pool(pool_hparams)
    cnn2 = CNN(conv_hparams2)
    relu2 = Relu()
    pool2 = Pool(pool_hparams)
    fc = FC(fully_con_hparams)
    relu3 = Relu()
    fc2 = FC(fully_con_hparams2)
    sigmoid = Sigmoid()

    model = [cnn, relu, pool, cnn2, relu2, pool2, fc, relu3, fc2, sigmoid]

    previous = train_X
    for module in model:
        if isinstance(module, Layer):
            previous = module.initialize_matrices(previous)

    for i in range(100):
        print("Loop: " + str(i + 1))
        print("")
        print("CNN 1 in progress.")
        Z1 = cnn.forward(train_X)
        A1 = relu.forward(Z1)
        P1 = pool.forward(A1)
        print("CNN 2 in progress.")
        Z2 = cnn2.forward(P1)
        A2 = relu2.forward(Z2)
        P2 = pool2.forward(A2)
        print("FCs in progress.")
        Z3 = fc.forward(P2)
        A3 = relu3.forward(Z3)
        Z4  = fc2.forward(A3)
        Y_pred = sigmoid.forward(Z4)
        J, error = log_loss(Y_pred, train_Y)
        print("---------------------")
        print("LOSS: " + str(J))
        print("ERROR: " + str(error))
        print("---------------------")
        dZ = Y_pred - train_Y
        print("")
        print("Backprop in progress.")
        fc2.backward(dZ)
        relu3.backward(fc2.dx)
        fc.backward(relu3.dx)
        pool2.backward(fc.dx)
        relu2.backward(pool2.dx)
        print("CNN backprop in progress.")
        cnn2.backward(relu2.dx)
        pool.backward(cnn2.dx)
        relu.backward(pool.dx)
        cnn.backward(relu.dx)
        for module in model:
            if isinstance(module, Layer) and module.has_weights:
                module.update_weights(learning_rate)

    # flatten = flatten(P1)
    # Z2 = fc.forward(flatten)
    # Y_hat = sigmoid(Z2)
    # J = loss(Y_hat, train_Y)
    # dZ4 = Y_hat - train_Y
    # dZ2 = fc.backward(dZ4, Z2, "relu")
