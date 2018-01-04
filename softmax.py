#!/usr/bin/env python

import numpy as np

"""
TODO: Better check to identify a matrix or a vector
TODO: Add more complex test
TODO: Check assertion error in the current test
TODO: Implement softmax gradient
TODO: Test softmax gradient
"""

def softmax(x):
    """Compute the softmax function for each row of the input x.
    For more information you can see: http://cs231n.github.io/neural-networks-case-study/.

    Arguments:
    x -- A N dimensional vector or M x N dimensional numpy matrix.

    Return:
    x -- You are allowed to modify x in-place
    """

    orig_shape = x.shape

    # Softmax implementation for matrix. I need to apply the algorithm for each row (axis=1)
    if len(x.shape) > 1:
        max_matrix = np.max(x, axis=1) # Get the max values according to the row

        # Reshape matrix from a vector of 2 elements to a matrix 2x1
        # max_matrix is 2x1
        max_matrix = max_matrix.reshape(max_matrix.shape[0], 1)

        # Numerical stability for performance improvement:
        # I subtract each element of the row with the
        # corresponding row (a single element) of the max vector
        x = x - max_matrix
        e = np.exp(x)
        x = np.divide(e, np.sum(e, axis=0))

    # Softmax implementation for vector.
    else:
        vector_max_value = np.max(x)
        x = (np.exp(x - vector_max_value)) / sum(np.exp(x - vector_max_value))

    assert x.shape == orig_shape
    return x


def softmax_grad(x):
    # To understand the softmax gradient see the following link:
    # https://eli.thegreenplace.net/2016/the-softmax-function-and-its-derivative/
    raise NotImplementedError


def test_softmax_and_its_gradient():
    print("Running basic tests...")
    test1 = softmax(np.array([1,2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2 = softmax(np.array([[1001,1002],[3,4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3 = softmax(np.array([[-1001,-1002]]))
    print(test3)
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_softmax_and_its_gradient()
