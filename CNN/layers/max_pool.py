from others.helpers import *
from others.base_classes import Layer

class Pool(Layer):
    def __init__(self, hparameters):
        self.f = hparameters["f"] # filter size
        self.pad = hparameters["pad"]
        self.stride = hparameters["stride"]
        self.output_width = None
        self.output_height = None
        self.has_weights = False

    def forward(self, A_previous):
        self.cache = A_previous
        for i in range(A_previous.shape[0]):
            a_prev = A_previous[i]
            for h in range(self.output_height):
                vert_start = h * self.stride
                vert_end = vert_start + self.f
                for w in range(self.output_width):
                    horiz_start = w * self.stride
                    horiz_end = horiz_start + self.f
                    for c in range(self.channels):
                        a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                        self.forward_out[i, h, w, c] = np.max(a_prev_slice)
        return self.forward_out

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

                        self.backward_out[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * da
        return self.backward_out

    def initialize_matrices(self, A_previous):
        (m, height_prev, width_prev, channels_prev) = np.shape(A_previous)
        self.get_output_size(height_prev, width_prev)
        self.channels = channels_prev
        self.forward_out = np.zeros((m, self.output_height, self.output_width, self.channels))
        self.backward_out = np.zeros_like(A_previous)
        return self.forward_out

    def get_output_size(self, height_prev, width_prev):
        self.output_height= int(1 + (height_prev - self.f) / self.stride)
        self.output_width = int(1 + (width_prev - self.f) / self.stride)
