#!/usr/bin/env python

import numpy as np
import random

def softmax(x):
    """Compute the softmax function for each row of the input x.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """

    orig_shape = x.shape # Get the shape of the matrix

    # Softmax implementation for matrix. I need to apply the algorithm for each row (axis=1)
    if len(x.shape) > 1:
        max_matrix = np.max(x, axis=1) # Get the max values according to the row

        # Reshape matrix from a vector of 2 elements to a matrix 2x1
        # TIPS: in genearal it is better to work with matrix (also when they represent vectors)
        # I calculate the max value because, in the next step, I will use it to improve the performanc
        # max_matrix is 2x1
        max_matrix = max_matrix.reshape(max_matrix.shape[0], 1)

        # I subtract each element of the row with each row (a single element) of the max vector
        x = x - max_matrix

        e = np.exp(x)

        # Softmax implementation
        x = np.divide(e, np.sum(e, axis=0))


    # Softmax implementation for vector. TODO: it can be improved with a better check
    else:
        # I calculate the max value because, in the next step, I will use it to improve the performance
        vector_max_value = np.max(x)

        # Softmax is invariant to constant offsets. I make this operation for numerical stability
        x = (np.exp(x - vector_max_value)) / sum(np.exp(x - vector_max_value))

def sigmoid(x):
    """
    Compute the sigmoid function for the input here.

    Arguments:
    x -- A scalar or numpy array.

    Return:
    s -- sigmoid(x)
    """

    s = 1. / (1. + np.exp(-x))
    return s

def sigmoid_grad(s):
    """
    Compute the gradient for the sigmoid function here. Note that
    for this implementation, the input s should be the sigmoid
    function value of your original input x.

    Arguments:
    s -- A scalar or numpy array.

    Return:
    ds -- Your computed gradient.
    """

    ds = s * (1 - s) # It is calculated exploiting the chain rule of derivation

    return ds
