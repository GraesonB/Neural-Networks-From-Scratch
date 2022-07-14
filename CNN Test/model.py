from helpers import *
from base_classes import *
from modules import *
from PIL import Image

class Model:
    def __init__(self, modules, train, hparameters):
        self.learning_rate = hparameters["learning_rate"]
        self.epochs = hparameters["epochs"]
        self.batch_size = hparameters["batch_size"]
        self.beta_1 = hparameters["beta_1"]
        self.beta_2 = hparameters["beta_2"]

        self.modules = modules
        self.number_of_mods = len(modules)
        self.train_X = train[0] / 255
        self.train_Y = train[1]
        self.total_batches = int(np.ceil(len(self.train_X) / self.batch_size))
        self.batch_number = 0
        self.batch_start, self.batch_end = 0, self.batch_size

    def run_model(self):
        self.initialize_model()
        for i in range(self.epochs):
            print("-------")
            print("EPOCH " + str(i + 1))
            print("-------")
            for batch_number in range(self.total_batches - 1):
                print("")
                print("BATCH: " + str(batch_number + 1) + "/" + str(self.total_batches - 1))
                print("")
                self.batch_start, self.batch_end = self.get_batch(batch_number)
                Y_pred = self.model_forward()
                J, acc = self.calculate_loss(Y_pred)

                self.model_backward(Y_pred)
                self.update_weights()

    def initialize_model(self):
        previous = self.train_X[self.batch_start:self.batch_end]
        for module in self.modules:
            if isinstance(module, Layer):
                previous = module.initialize_matrices(previous)

    def model_forward(self):
        previous_out = self.train_X[self.batch_start:self.batch_end]
        for idx, module in enumerate(self.modules):
            print("Forward in progress: " + str(idx + 1) + "/" + str(self.number_of_mods))
            previous_out = module.forward(previous_out)
        Y_pred = previous_out
        return Y_pred

    def model_backward(self, Y_pred):
        dout = Y_pred - self.train_Y[self.batch_start:self.batch_end]
        # iterate through modules in reverse order
        reversed_modules = list(reversed(self.modules))
        for idx, module in enumerate(reversed_modules):
            print("Backward in progress: " + str(idx + 1) + "/" + str(self.number_of_mods))
            if idx == 0:
                module.backward(dout)
                continue
            dout = reversed_modules[idx - 1].dx
            module.backward(dout)

    def calculate_loss(self, Y_pred):
        J, acc = binary_cross_entropy_loss(Y_pred, self.train_Y[self.batch_start:self.batch_end])
        print("------------------------")
        print("LOSS: " + str(J))
        print("ACCURACY: " + str(acc))
        print("------------------------")
        return J, acc

    def update_weights(self):
        # loop through all non-actviation/non-pooling layers and update weights
        for module in self.modules:
            if isinstance(module, Layer) and module.has_weights:
                module.update_weights(self.learning_rate, self.beta_1, self.beta_2)

    def get_batch(self, batch_number):
        # when the batch size doesn't evenly divide into dataset, need to handle the last batch
        if (batch_number * self.batch_size + self.batch_size) > len(self.train_X):
            batch_start = batch_number * self.batch_size
            batch_end = len(self.train_X) - 1
            batch_size = batch_end - batch_start
        else:
            batch_start = batch_number * self.batch_size
            batch_end = batch_start + self.batch_size
            batch_size = self.batch_size

        return batch_start, batch_end
