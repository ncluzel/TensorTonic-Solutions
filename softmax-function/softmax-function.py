import numpy as np

def softmax(x):
    """
    Compute the softmax of input x.
    Works for 1D or 2D NumPy arrays.
    For 2D, compute row-wise softmax.
    """
    axis = 1
    x = np.array(x)
    if len(x.shape)==1:
        axis = 0
    else:
        if x.shape[1]==1:
            axis = 0
    # logsumexp trick
    x_max = np.max(x)
    num = np.exp(x - x_max)
    return num / np.sum(num, axis=axis, keepdims=True)