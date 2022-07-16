from others.helpers import *
from others.base_classes import Layer

class ConvLayer(Layer):
    def __init__(self, hparameters):
        super().__init__()
        self.channels = hparameters["channels"]
        self.f = hparameters["f"] # filter size
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]

        self.output_width = None
        self.output_height = None

        self.has_weights = True
        self.time = 0

    def forward(self, A_previous):
        self.cache = A_previous
        # pad A_previous
        A_previous_pad = zero_pad(A_previous, self.pad)
        # the following for loop gets the output one value at a time.
        for i in range(A_previous.shape[0]):
            a_previous_pad = A_previous_pad[i]
            for h in range(self.output_height):
                for w in range(self.output_width):
                    for c in range(self.channels):
                        # define the box
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_slice_prev = a_previous_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = self.weights[:,:,:,c]
                        biases = self.biases[:,:,:,c]
                        self.forward_out[i, h, w, c] = single_convolution(a_slice_prev, weights, biases)
        # cache for backprop
        return self.forward_out

    def backward(self, dout):
        # get dout dims
        (m, height, width, channels) = np.shape(dout)
        # pad time
        A_previous_pad = zero_pad(self.cache, self.pad)
        dA_previous_pad = zero_pad(self.backward_out, self.pad)
        #
        for i in range(m):
            a_previous_pad = A_previous_pad[i]
            da_previous_pad = dA_previous_pad[i]
            for h in range(height):
                for w in range(width):
                    for c in range (channels):
                        # define box
                        vert_start = h * self.stride
                        vert_end = vert_start + self.f
                        horiz_start = w * self.stride
                        horiz_end = horiz_start + self.f

                        a_slice = a_previous_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        da_previous_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.weights[:,:,:,c] * dout[i,h,w,c]
                        self.weight_grads[:,:,:,c] += a_slice * dout[i, h, w, c]
                        self.bias_grads[:,:,:,c] += dout[i, h, w, c]
            # remove pad and add ith example to dA_previous
            if self.pad != 0:
                self.backward_out[i,:,:,:] = da_previous_pad[self.pad:-self.pad, self.pad:-self.pad, :]
            else:
                self.backward_out[i,:,:,:] = da_previous_pad
        return self.backward_out

    # initializes weight matrices and output shapes
    def initialize_matrices(self, A_previous):
        (m, height_prev, width_prev, channels_prev) = np.shape(A_previous)
        self.get_output_size(height_prev, width_prev)
        self.initialize_weights(height_prev, width_prev, channels_prev)
        self.backward_out = np.zeros_like(A_previous)
        self.forward_out = np.zeros((m, self.output_height, self.output_height, self.channels))
        return self.forward_out

    def get_output_size(self, height_prev, width_prev):
        self.output_width = int(((height_prev - self.f + (2*self.pad)) / self.stride)) + 1
        self.output_height = int(((width_prev - self.f + (2*self.pad)) / self.stride)) + 1

    def initialize_weights(self, height_prev, width_prev, channels_prev):
        n = ((height_prev * width_prev * channels_prev) + 1) * self.channels
        self.weights = np.random.rand(self.f, self.f, channels_prev, self.channels) * np.sqrt(2 / n)
        self.biases = np.zeros((1,1,1, self.channels))
        self.weight_grads = np.zeros(((self.f, self.f, channels_prev, self.channels)))
        self.bias_grads = np.zeros((1,1,1,self.channels))
        self.SdW = np.zeros_like(self.weight_grads)
        self.Sdb = np.zeros_like(self.bias_grads)
        self.VdW = np.zeros_like(self.weight_grads)
        self.Vdb = np.zeros_like(self.bias_grads)
