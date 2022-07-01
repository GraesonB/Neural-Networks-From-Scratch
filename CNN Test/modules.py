from helpers import *
from base_classes import *
import datasets as data


class Relu(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z
        return relu(Z)

    def backward(self, dA):
        self.dx = (self.cache >= 0)
        self.dx = self.dx * dA

class Sigmoid(Activation):
    def __init__(self):
        super().__init__()

    def forward(self, Z):
        self.cache = Z

        return sigmoid_f(Z)

    def backward(self, dA):
        self.dx = sigmoid_derivative(dA)

class FC(Layer):
    def __init__(self, hparameters):
        super().__init__()
        self.nodes = hparameters["nodes"]

        self.has_weights = True

    def forward(self, A_prev):
        self.cache = A_prev
        if A_prev.ndim != 2:
            A_prev = flatten(A_prev)
        Z = np.dot(A_prev, self.W) + self.b
        self.flattened_cache = A_prev
        return Z

    def backward(self, dout):
        m = len(self.cache[0])
        self.dW = np.dot(self.flattened_cache.T, dout)
        # keep dims for db to avoid creating rank-1 array
        self.db = np.sum(dout, axis = 0, keepdims = True)
        self.dx = np.dot(dout, self.W.T)
        self.dx = np.reshape(self.dx, self.cache.shape)

    # TODO
    def initialize_matrices(self, A_prev):
        if A_prev.ndim != 2:
            A_prev = flatten(A_prev)
        self.W = np.random.rand(A_prev.shape[1], self.nodes) * 0.001
        self.b = np.zeros((1,self.nodes)) * 0.01
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
        self.initialize_weights(channels_prev)
        self.dx = np.zeros_like(A_prev)
        self.Z = np.zeros((m, self.output_height, self.output_height, self.channels))
        return self.Z

    def get_output_size(self, height_prev, width_prev):
        self.output_width = int(((height_prev - self.f + (2*self.pad)) / self.stride)) + 1
        self.output_height = int(((width_prev - self.f + (2*self.pad)) / self.stride)) + 1

    def initialize_weights(self, channels_prev):
        self.W = np.random.rand(self.f, self.f, channels_prev, self.channels) * 0.001
        self.b = np.zeros((1,1,1, self.channels)) * 0.01
        self.dW = np.zeros(((self.f, self.f, channels_prev, self.channels)))
        self.db = np.zeros((1,1,1,self.channels))

class Pool(Layer):
    def __init__(self, hparameters):
        self.f = hparameters["f"]
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]

        self.output_width = None
        self.output_height = None
        self.A = None

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
