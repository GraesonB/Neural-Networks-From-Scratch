import numpy as np
from PIL import Image

import datasets as data
from model import Model
from others.base_classes import *
from others.flatten import Flatten
from layers.activations import Relu, Sigmoid
from layers.conv import ConvLayer
from layers.fully_connected import FCLayer
from layers.max_pool import Pool
from layers.batch_norm import BatchNorm

# Hyperparams -----------------------------------------------------------------#
conv_hparams = {
"channels" : 6,
"f" : 5, # filter size
"stride" : 2,
"pad" : 1
}

conv_hparams2 = {
"channels" : 10,
"f" : 5,
"stride" : 2,
"pad" : 1
}

conv_hparams3 = {
"channels" : 10,
"f" : 5,
"stride" : 2,
"pad" : 0
}

pool_hparams = {
"stride" : 2,
"pad" : 0,
"f" : 2
}

fully_con_hparams = {
"nodes" : 256
}

fully_con_hparams2 = {
"nodes" : 128
}

fully_con_hparams_f = {
"nodes" : 1
}

model_hparams = {
"learning_rate" : 0.0001,
"epochs" : 10,
"batch_size" : 64,
"beta_1" : 0.9,
"beta_2" : 0.999,
}

# Program ---------------------------------------------------------------------#

if __name__ == "__main__":
    print("Loading data...")
    train, test, _ = data.load_cats_dogs_1k()
    train[1] = np.reshape(train[1], (-1, 1))
    test[1] = np.reshape(test[1], (-1, 1))
    print("Building model...")

    cnn = ConvLayer(conv_hparams)
    cnn2 = ConvLayer(conv_hparams2)

    pool = Pool(pool_hparams)
    pool2 = Pool(pool_hparams)

    batch_norm = BatchNorm()
    batch_norm2 = BatchNorm()
    batch_norm3 = BatchNorm()
    batch_norm4 = BatchNorm()

    relu = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()

    fc = FCLayer(fully_con_hparams)
    fc2 = FCLayer(fully_con_hparams)
    fc_final = FCLayer(fully_con_hparams_f)

    sigmoid = Sigmoid()
    flatten = Flatten()

    modules = [cnn, batch_norm, relu, pool, cnn2, batch_norm2, relu2, flatten,
    fc, batch_norm3, relu3, fc2, batch_norm4, relu4, fc_final, sigmoid]

    cnn_model = Model(modules, train, test, model_hparams)

    cnn_model.run_model()
