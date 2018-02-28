#!/usr/bin/env python

import numpy as np

"""
TODO: Better check to identify a matrix or a vector
TODO: Implement softmax gradient for vector
TODO: Check if its better have a vector (n,m) or (m,n)
"""

def softmax(Z):
    """
    Arguments:
    Z -- numpy array of shape (n,m) where n is the number of features while m is the number of samples
    In other words, each coloumn is a sample of data

    Return:
    A -- Output of the softmax function
    cache -- returns Z as well, useful during backpropagation
    """

    # Softmax implementation for matrix. I need to apply the algorithm for each coloumn (axis=0)
    if len(Z.shape) > 1:
        max_matrix = np.max(Z, axis=0)
        stable_Z = Z - max_matrix
        e = np.exp(stable_Z)
        A = e / np.sum(e, axis=0, keepdims=True)
    # Softmax implementation for vector.
    else:
        vector_max_value = np.max(Z)
        A = (np.exp(Z - vector_max_value)) / sum(np.exp(Z - vector_max_value))

    assert A.shape == Z.shape

    cache = Z

    return A, cache


def softmax_grad(dA, cache):
    """
    Arguments:
    dA -- post-activation gradient
    cache -- 'Z' where we store for computing backward propagation efficiently

    Return:
    dZ -- Jacobian Matrix

    np.array([D1S1,...,D1SN],
             [D2S1,...,D2SN],
             ...,
             [D1SN,...,DNSN])

    Si(1-Sj) i=j
    -SjSi i different from j

    """
    Z = cache
    s = softmax(Z)[0].reshape(-1, 1)
    dZ = dA * (np.diagflat(s) - np.dot(s, s.T))

    assert dZ.shape[0] == (Z.shape[0] * Z.shape[1])

    return dZ


def test_softmax_and_its_gradient():
    print("Running softmax() and softmax_grad() tests...")

    test1, cache = softmax(np.array([1,2]))
    print(test1)
    ans1 = np.array([0.26894142,  0.73105858])

    assert np.allclose(test1, ans1, rtol=1e-05, atol=1e-06)

    test2, cache = softmax(np.array([[1001,3],[1002,4]]))
    print(test2)
    ans2 = np.array([
        [0.26894142, 0.26894142],
        [0.73105858, 0.73105858]])

    assert np.allclose(test2, ans2, rtol=1e-05, atol=1e-06)

    #test1_grad = softmax_grad(np.array([1,2]))
    #print(test1_grad)

    test2_grad = softmax_grad(1, np.array([[1001,3],[1002,4]]))
    print(test2_grad)
    anstest2_grad = np.array([
        [ 0.19661193,-0.07232949,-0.19661193,-0.19661193],
        [-0.07232949,0.19661193,-0.19661193,-0.19661193],
        [-0.19661193,-0.19661193,0.19661193,-0.53444665],
        [-0.19661193,-0.19661193,-0.53444665,0.19661193]])

    assert np.allclose(test2_grad, anstest2_grad, rtol=1e-05, atol=1e-06)

    print("... end tests")

if __name__ == "__main__":
    test_softmax_and_its_gradient()
