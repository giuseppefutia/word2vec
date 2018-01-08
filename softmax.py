#!/usr/bin/env python

import numpy as np

"""
TODO: Better check to identify a matrix or a vector
TODO: Implement softmax gradient for vector
"""

def softmax(Z):
    """
    Arguments:
    Z -- numpy array of any shape

    Return:
    A -- Output of the softmax function
    cache -- returns Z as well, useful during backpropagation
    """

    # Softmax implementation for matrix. I need to apply the algorithm for each row (axis=1)
    if len(Z.shape) > 1:
        max_matrix = np.max(Z, axis=1)
        max_matrix = max_matrix.reshape(max_matrix.shape[0], 1)
        stable_Z = Z - max_matrix
        e = np.exp(stable_Z)
        A = np.divide(e, np.sum(e, axis=1, keepdims=True))
    # Softmax implementation for vector.
    else:
        vector_max_value = np.max(Z)
        A = (np.exp(Z - vector_max_value)) / sum(np.exp(Z - vector_max_value))

    assert A.shape == Z.shape

    cache = Z

    return A, cache


def softmax_grad(cache):
    Z = cache
    s = softmax(Z)[0].reshape(-1, 1)
    dZ = np.diagflat(s) - np.dot(s, s.T)

    assert dZ.shape[0] == (Z.shape[0] * Z.shape[1])

    return dZ


def test_softmax_and_its_gradient():
    print("Running basic tests...")

    test1, cache = softmax(np.array([1,2]))
    ans1 = np.array([0.26894142,  0.73105858])
    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2, cache = softmax(np.array([[1001,1002],[3,4]]))
    ans2 = np.array([
        [0.26894142, 0.73105858],
        [0.26894142, 0.73105858]])
    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    test3, cache = softmax(np.array([[-1001,-1002]]))
    ans3 = np.array([0.73105858, 0.26894142])
    assert np.allclose(test3, ans3, rtol=1e-05, atol=1e-06)

    #test1_grad = softmax_grad(np.array([1,2]))
    #print(test1_grad)

    test2_grad = softmax_grad(np.array([[1001,1002,1003],[3,4,5]]))
    anstest2_grad = np.array([
    [ 0.08192507, -0.02203304, -0.05989202, -0.0081055,  -0.02203304, -0.05989202,],
    [-0.02203304,  0.18483645, -0.1628034,  -0.02203304, -0.05989202, -0.1628034, ],
    [-0.05989202, -0.1628034,   0.22269543, -0.05989202, -0.1628034,  -0.44254553,],
    [-0.0081055,  -0.02203304, -0.05989202,  0.08192507, -0.02203304, -0.05989202,],
    [-0.02203304, -0.05989202, -0.1628034,  -0.02203304,  0.18483645, -0.1628034, ],
    [-0.05989202, -0.1628034,  -0.44254553, -0.05989202, -0.1628034,   0.22269543,],
    ])

    assert np.allclose(test2_grad, anstest2_grad, rtol=1e-05, atol=1e-06)

    print("... test OK!")

if __name__ == "__main__":
    test_softmax_and_its_gradient()
