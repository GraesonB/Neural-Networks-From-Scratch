import numpy as np
from PIL import Image

from model import Model
from base_classes import *
from modules import *
import datasets as data
from function_layers import *
from mlxtend.data import loadlocal_mnist

# Hyperparams -----------------------------------------------------------------#
conv_hparams = {
"channels" : 6,
"f" : 5,
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
"epochs" : 100,
"batch_size" : 64,
"beta_1" : 0.9,
"beta_2" : 0.999,
}
# when loss doesn't improve let the learning rate decrease
# loss doesn't always converge to 0 specieally in image dataset. It could be like 30!!
# check with your validation set to see whether your model is overfitting or underfitting
# Things to implement: setting right initialization, batch_norm, learning rate scheduling
# learning rate scheduling : cosine learning rate scheduler, learning rate decay, schedule learning rate decrease
np.random.seed(0)

# Program ---------------------------------------------------------------------#

if __name__ == "__main__":
    print("Loading data...")
    train, _, _ = data.load_cats_dogs_25k()
    train[1] = np.reshape(train[1], (-1, 1))
    print("Building model...")

    cnn = CNN(conv_hparams)
    cnn2 = CNN(conv_hparams2)

    pool = Pool(pool_hparams)
    pool2 = Pool(pool_hparams)

    batch_norm = BatchNorm()
    batch_norm2 = BatchNorm()
    batch_norm3 = BatchNorm()
    batch_norm4 = BatchNorm()
    batch_norm5 = BatchNorm()


    relu = Relu()
    relu2 = Relu()
    relu3 = Relu()
    relu4 = Relu()
    relu5 = Relu()
    relu6 = Relu()
    relu7 = Relu()

    fc = FC(fully_con_hparams)
    fc2 = FC(fully_con_hparams)
    fc3 = FC(fully_con_hparams)
    fc4 = FC(fully_con_hparams2)
    fc5 = FC(fully_con_hparams2)
    fc6 = FC(fully_con_hparams2)
    fc7 = FC(fully_con_hparams2)
    fc_final = FC(fully_con_hparams_f)

    sigmoid = Sigmoid()
    flatten = Flatten()

    modules = [cnn, batch_norm, relu, pool, cnn2, batch_norm5, relu5, flatten, 
    fc, batch_norm2, relu2, fc2, batch_norm3, relu3, fc_final, sigmoid]

    cnn_model = Model(modules, train, model_hparams)

    cnn_model.run_model()
