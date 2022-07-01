from helpers import *
from base_classes import *

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
                        print(self.db)
            # remove pad and add ith example to dA_prev
            self.dx[i,:,:,:] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]

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
        self.W = np.random.rand(self.f, self.f, channels_prev, self.channels) * 0.01
        self.b = np.random.rand(1,1,1, self.channels) * 0.01
        self.dW = np.zeros(((self.f, self.f, channels_prev, self.channels)))
        self.db = np.zeros((1,1,1,self.channels))


if __name__ == "__main__":
    pass
