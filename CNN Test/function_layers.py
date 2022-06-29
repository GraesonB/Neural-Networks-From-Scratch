import numpy as np

# FORWARD FUNCTIONS
def conv_forward(A_prev, W, b, hparameters):
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)

    # get hyperparameters
    stride = hparameters['stride']
    pad = hparameters['pad']

    # calculate dims of output, int for flooring
    n_H = int(((n_H_prev - f + (2*pad)) / stride)) + 1
    n_W = int(((n_W_prev - f + (2*pad)) / stride)) + 1

    # initialize output
    Z = np.zeros((m, n_H, n_W, n_C))
    A_prev_pad = zero_pad(A_prev, pad)

    # the following for loop gets the output one value at a time.
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_slice_prev = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]
                    weights = W[:,:,:,c]
                    biases = b[:,:,:,c]
                    Z[i, h, w, c] = single_convolution(a_slice_prev, weights, biases)

    # cache for backprop
    cache = (A_prev, W, b, hparameters)
    return Z, cache

def pool_forward(A_prev, hparameters):
    # Retrieve dimensions from the input shape
    (m, n_H_prev, n_W_prev, n_C_prev) = A_prev.shape

    # get hyperparameters
    f = hparameters["f"]
    stride = hparameters["stride"]

    # calculate output dims
    n_H = int(1 + (n_H_prev - f) / stride)
    n_W = int(1 + (n_W_prev - f) / stride)
    n_C = n_C_prev

    # Initialize output
    A = np.zeros((m, n_H, n_W, n_C))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    A[i, h, w, c] = np.max(a_prev_slice)
    print("Maxed.")
    print("-----------------------------------------------")

    # Store the input and hparameters in "cache" for pool_backward()
    cache = (A_prev, hparameters)
    return A, cache

# BACKWARD FUNCTIONS
def conv_backward(dZ, cache):
    # unpack cache
    (A_prev, W, b, hparameters) = cache
    (m, n_H_prev, n_W_prev, n_C_prev) = np.shape(A_prev)
    (f, f, n_C_prev, n_C) = np.shape(W)

    # get hparams
    stride = hparameters['stride']
    pad = hparameters['pad']

    # get dZ dims
    (m, n_H, n_W, n_C) = np.shape(dZ)

    # initialize outputs
    dA_prev = np.zeros((m, n_H_prev, n_W_prev, n_C_prev))
    dW = np.zeros(((f, f, n_C_prev, n_C)))
    db = np.zeros((1,1,1,n_C))

    # pad time
    A_prev_pad = zero_pad(A_prev, pad)
    dA_prev_pad = zero_pad(dA_prev, pad)

    # review how CNN back prop works
    for i in range(m):
        a_prev_pad = A_prev_pad[i]
        da_prev_pad = dA_prev_pad[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range (n_C):
                    a_slice = a_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :]

                    da_prev_pad[vert_start:vert_end, horiz_start:horiz_end, :] += W[:,:,:,c] * dZ[i,h,w,c]
                    dW[:,:,:,c] += a_slice * dZ[i, h, w, c]
                    db[:,:,:,c] += dZ[i, h, w, c]
        # remove pad and add ith example to dA_prev
        dA_prev[i,:,:,:] = da_prev_pad[pad:-pad, pad:-pad, :]

    return dA_prev, dW, db

def pool_backward(dA, cache):
    # unpack cache
    (A_prev, hparameters) = cache

    # get hparams
    stride = hparameters['stride']
    f = hparameters['f']

    # get shapes
    m, n_H_prev, n_W_prev, n_C_prev = np.shape(A_prev)
    m, n_H, n_W, n_C = np.shape(dA)

    # initialize output shape
    dA_prev = np.zeros(np.shape(A_prev))

    for i in range(m):
        a_prev = A_prev[i]
        for h in range(n_H):
            vert_start = h * stride
            vert_end = vert_start + f
            for w in range(n_W):
                horiz_start = w * stride
                horiz_end = horiz_start + f
                for c in range(n_C):
                    a_prev_slice = a_prev[vert_start:vert_end, horiz_start:horiz_end, c]
                    da = dA[i,h,w,c]
                    mask = create_mask_from_window(a_prev_slice)

                    dA_prev[i, vert_start: vert_end, horiz_start: horiz_end, c] += mask * da

    return dA_prev
