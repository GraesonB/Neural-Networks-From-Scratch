from helpers import *

class FC:
    def __init__(self, hparameters):
        self.nodes = hparameters["nodes"]
        self.cached_A_prev = None
        self.Z = None
        self.W = None
        self.b = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        if self.W == None:
            self.initalize_matrices(A_prev)
        self.Z = np.dot(A_prev, self.W) + self.b
        self.cached_A_prev = A_prev
        return self.Z

    def backward(self, dZ, prev_Z, activation):
        m = len(self.cached_A_prev[0])
        self.dW = 1/m * np.dot(self.cached_A_prev.T, dZ)
        self.db = 1/m * np.sum(dZ, axis = 0, keepdims = True)

        if activation == "relu":
            prev_dZ = np.dot(dZ, self.W.T) * sigmoid_derivative(prev_Z)
        return prev_dZ

    # TODO
    def initalize_matrices(self, A_prev):
        self.W = np.random.rand(A_prev.shape[1], self.nodes) * 0.01
        self.b = np.random.rand(1,self.nodes) * 0.01

class CNN:
    def __init__(self, hparameters):
        self.channels = hparameters["channels"]
        self.f = hparameters["f"]
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]

        self.cached_A_prev = None
        self.output_width = None
        self.output_height = None
        self.W = None
        self.b = None
        self.Z = None
        self.dW = None
        self.db = None

    def forward(self, A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)

        if self.output_width == None:
            self.get_output_size(n_H_prev, n_W_prev)
        if self.W == None:
            self.initialize_weights(n_C_prev)
        # initialize output
        if self.Z == None:
            self.Z = np.zeros((m, self.output_height, self.output_height, self.channels))
        A_prev_pad = zero_pad(A_prev, self.pad)

        # the following for loop gets the output one value at a time.
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            for h in range(self.output_height):
                vert_start = h * self.stride
                vert_end = vert_start + self.f
                for w in range(self.output_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.f
                    for c in range(self.channels):
                        a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                        weights = self.W[:,:,:,c]
                        biases = self.b[:,:,:,c]
                        self.Z[i, h, w, c] = single_convolution(a_slice_prev, weights, biases)

        # cache for backprop
        self.cached_A_prev = A_prev
        print("Finished forward pass on " + str(m) + " images.")
        print("-----------------------------------------------")
        return self.Z

    def backward(self, dA):
        (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(self.cached_A_prev)

        # get dZ dims
        (m, n_H, n_W, n_C) = np.shape(dZ)
        # initialize output
        self.dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))

        # pad time
        A_prev_pad = zero_pad(self.cached_A_prev, self.pad)
        dA_prev_pad = zero_pad(dA_prev, self.pad)

        # review how CNN back prop works
        for i in range(m):
            a_prev_pad = A_prev_pad[i]
            da_prev_pad = dA_prev_pad[i]
            for h in range(self.output_height):
                vert_start = h * stride
                vert_end = vert_start + f
                for w in range(self.output_width):
                    horiz_start = w * stride
                    horiz_end = horiz_start + f
                    for c in range (self.channels):
                        a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                        da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += self.W[:,:,:,c] * dZ[i,h,w,c]
                        self.dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                        self.db[:,:,:,c] += dZ[i, h, w, c]
            # remove pad and add ith example to dA_prev
            self.dA_prev[i,:,:,:] = da_prev_pad[self.pad:-self.pad, self.pad:-self.pad, :]

    #TODO
    def initialize_matrices(self, A_prev):
        (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
        self.get_output_size(n_H_prev, n_W_prev)
        self.initialize_weights(n_C_prev)

    def get_output_size(self, n_H_prev, n_W_prev):
        self.output_width = int(((n_H_prev - self.f + (2*self.pad)) / self.stride)) + 1
        self.output_height = int(((n_W_prev - self.f + (2*self.pad)) / self.stride)) + 1

    def initialize_weights(self, n_C_prev):
        self.W = np.random.rand(self.f, self.f, n_C_prev, self.channels) * 0.01
        self.b = np.random.rand(1,1,1, self.channels) * 0.01
        self.dW = np.zeros(((self.f, self.f, n_C_prev, self.channels)))
        self.db = np.zeros((1,1,1,n_C_prev))
