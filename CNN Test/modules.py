from helpers import *
from base_classes import *
import datasets as data
from numba import jit, cuda


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        return np.where(Z >= 0, Z, 0)

    def backward(self, dA):
        self.dx = (self.cache >= 0)
        self.dx = dA * np.where(self.cache >= 0, 1, 0)

class Sigmoid(Activation):

    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        self.output = sigmoid_f(Z)
        return self.output

    def backward(self, dA):
        yup = sigmoid_f(self.cache)
        self.dx = np.multiply(np.multiply(self.output, (1 - self.output)), dA)


class Softmax(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        Z_ = Z - Z.max()
        e = np.exp(Z_)
        return e / np.sum(e, axis=0, keepdims=True)

    def backward(self, dA):
        return dA * (self.cache * (1 - self.cache))

class BatchNorm(Layer):
    def __init__(self):
        self.time = 0
        self.has_weights = True
        self.gamma = None
        self.beta = None
        self.cache = None
        self.z_norm = None
        self.forward_out = None
        self.d_gamma = None
        self.d_beta = None
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
        self.d_beta = np.sum(dA, axis = 0)
        self.d_gamma = np.sum((self.z_norm * dA), axis = 0)
        self.dx = (self.gamma * t / m) * (m * dA - np.sum(dA, axis=0)
         - t**2 * (self.cache-self.mean) * np.sum(dA*(self.cache - self.mean), axis=0))

    def initialize_matrices(self, A_prev):
        self.gamma = np.ones(A_prev.shape[1:])
        self.beta = np.zeros(A_prev.shape[1:])
        self.d_gamma = np.zeros_like(self.gamma)
        self.d_beta = np.zeros_like(self.gamma)
        self.Vd_gamma = np.zeros_like(self.gamma)
        self.Vd_beta = np.zeros_like(self.gamma)
        self.Sd_gamma = np.zeros_like(self.gamma)
        self.Sd_beta = np.zeros_like(self.gamma)
        self.forward_out = np.zeros_like(A_prev)
        return self.forward_out

    def update_weights(self, learning_rate, beta_1, beta_2, epsilon = 1e-8):
        self.time += 1
        # V terms are for GD with momentum, S terms are for RMS prop, as well as beta 1 and 2 respectively
        self.Vd_gamma = beta_1 * self.Vd_gamma + (1 - beta_1) * self.d_gamma
        self.Vd_beta = beta_1 * self.Vd_beta + (1 - beta_1) * self.d_beta
        self.Sd_gamma = beta_2 * self.Sd_gamma + (1 - beta_2) * self.d_gamma ** 2
        self.Sd_beta = beta_2 * self.Sd_beta + (1 - beta_2) * self.d_beta ** 2
        Vd_gamma_corrected = self.Vd_gamma / (1 - beta_1 ** self.time)
        Vd_beta_corrected = self.Vd_beta / (1 - beta_1 ** self.time)
        Sd_gamma_corrected = self.Sd_gamma / (1 - beta_2 ** self.time)
        Sd_beta_corrected = self.Sd_beta / (1 - beta_2 ** self.time)

        # self.W = self.W - learning_rate * self.dW
        # self.b = self.b - learning_rate * self.db
        self.gamma = self.gamma - (learning_rate * Vd_gamma_corrected / (np.sqrt(Sd_gamma_corrected) + epsilon))
        self.beta = self.beta - (learning_rate * Vd_beta_corrected / (np.sqrt(Sd_beta_corrected) + epsilon))

class Flatten:
    def __init__(self):
        self.shape = ()
        self.dx = None

    def forward(self, A):
        self.shape = A.shape
        data = np.ravel(A).reshape(self.shape[0], -1)
        return data

    def backward(self, dZ):
        self.dx = dZ.reshape(self.shape)

class FC(Layer):
    def __init__(self, hparameters):
        super().__init__()
        self.nodes = hparameters["nodes"]

        self.has_weights = True
        self.time = 0

    def forward(self, A_prev):
        self.cache = A_prev
        Z = np.dot(A_prev, self.W) + self.b
        return Z

    def backward(self, dout):
        m = len(self.cache[0])
        self.dW = (1/64) * np.dot(self.cache.T, dout)
        # keep dims for db to avoid creating rank-1 array
        self.db = (1/64) * np.sum(dout, axis = 0, keepdims = True)
        self.dx = np.dot(dout, self.W.T)

    def initialize_matrices(self, A_prev):
        if A_prev.ndim != 2:
            A_prev = flatten(A_prev)
        n = A_prev.shape[1] * self.nodes + 1
        self.W = he_uniform((A_prev.shape[1], self.nodes))
        self.b = np.zeros((1,self.nodes))
        self.SdW = np.zeros_like(self.W)
        self.Sdb = np.zeros_like(self.b)
        self.VdW = np.zeros_like(self.W)
        self.Vdb = np.zeros_like(self.b)
        self.Z = np.zeros((A_prev.shape[0], self.W.shape[1]))
        return self.Z

class CNN(Layer):
    def __init__(self, hparameters):
        super().__init__()
        self.channels = hparameters["channels"]
        self.f = hparameters["f"]
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]

        self.output_width = None
        self.output_height = None

        self.has_weights = True
        self.time = 0

    def forward(self, A_prev):
        self.cache = A_prev
        # pad A_prev
        A_prev_pad = zero_pad(A_prev, self.pad)
        # the following for loop gets the output one value at a time.
        for i in range(A_prev.shape[0]):
            a_prev_pad = A_prev_pad[i]
            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.channels):
                        # define the box
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = self.W[:,:,:,c]
                        biases = self.b[:,:,:,c]
                        self.Z[i, h, w, c] = single_convolution(a_slice_prev, weights, biases)
        # cache for backprop
        return self.Z

    def backward(self, dout):
        # get dZ dims
        (m, height, width, channels) = np.shape(dout)
        # pad time
        A_prev_pad = zero_pad(self.cache, self.pad)
        dA_prev_pad = zero_pad(self.dx, self.pad)

        # review how CNN back prop works
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(height):
                for w in range(width):
                    for c in range (channels):
                        # define box
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:,:,:,c] * dout[i,h,w,c]
                        self.dW[:,:,:,c] += a_slice * dout[i, h, w, c]
                        self.db[:,:,:,c] += dout[i, h, w, c]
            # remove pad and add ith example to dA_prev
            if self.pad != 0:
                self.dx[i,:,:,:] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]
            else:
                self.dx[i,:,:,:] = da_prev_pad

    def initialize_matrices(self, A_prev):
        (m, height_prev, width_prev, channels_prev) = np.shape(A_prev)
        self.get_output_size(height_prev, width_prev)
        self.initialize_weights(height_prev, width_prev, channels_prev)
        self.dx = np.zeros_like(A_prev)
        self.Z = np.zeros((m, self.output_height, self.output_height, self.channels))
        return self.Z

    def get_output_size(self, height_prev, width_prev):
        self.output_width = int(((height_prev - self.f + (2*self.pad)) / self.stride)) + 1
        self.output_height = int(((width_prev - self.f + (2*self.pad)) / self.stride)) + 1

    def initialize_weights(self, height_prev, width_prev, channels_prev):
        n = ((height_prev * width_prev * channels_prev) + 1) * self.channels
        self.W = np.random.rand(self.f, self.f, channels_prev, self.channels) * np.sqrt(2 / n)
        self.b = np.zeros((1,1,1, self.channels))
        self.dW = np.zeros(((self.f, self.f, channels_prev, self.channels)))
        self.db = np.zeros((1,1,1,self.channels))
        self.SdW = np.zeros_like(self.dW)
        self.Sdb = np.zeros_like(self.db)
        self.VdW = np.zeros_like(self.dW)
        self.Vdb = np.zeros_like(self.db)

class Pool(Layer):
    def __init__(self, hparameters):
        self.f = hparameters["f"]
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]

        self.output_width = None
        self.output_height = None

        self.has_weights = False

    def forward(self, A_prev):
        self.cache = A_prev
        for i in range(A_prev.shape[0]):
            a_prev = A_prev[i]
            for h in range(self.output_height):
                vert_start = h * self.stride
                vert_end = vert_start + self.f
                for w in range(self.output_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.f
                    for c in range(self.channels):
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        self.A[i, h, w, c] = np.max(a_prev_slice)

        # Store the input and hparameters in "cache" for pool_backward()
        return self.A

    def backward(self, dout):
        # get shape
        m, height, width, channels = np.shape(dout)

        for i in range(m):
            a_prev = self.cache[i]
            for h in range(height):
                for w in range(width):
                    for c in range(channels):
                        # define window
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        da = dout[i,h,w,c]
                        mask = create_mask_from_window(a_prev_slice)

                        self.dx[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * da

    def initialize_matrices(self, A_prev):
        (m, height_prev, width_prev, channels_prev) = np.shape(A_prev)
        self.get_output_size(height_prev, width_prev)
        self.channels = channels_prev
        self.A = np.zeros((m, self.output_height, self.output_width, self.channels))
        self.dx = np.zeros_like(A_prev)
        return self.A

    def get_output_size(self, height_prev, width_prev):
        self.output_height= int(1 + (height_prev - self.f) / self.stride)
        self.output_width = int(1 + (width_prev - self.f) / self.stride)


if __name__ == "__main__":
    train_X, train_Y = data.load_cats_dogs_64()
    train_X, train_Y = np.stack(train_X), np.reshape(train_Y, (-1, 1))

    pictures = train_X[0:2]
    cache = pictures
    print(pictures)
    pictures = flatten(pictures)
    print("---------------")
    print(pictures)
    pictures = np.reshape(pictures, cache.shape)
    print("---------------")
    print(pictures)
