#!/usr/bin/env python

import numpy as np

"""
TODO: update test implementation for sigmoid and sigmoid_grad
"""

def sigmoid(Z):
    """
    Arguments:
    Z -- numpy array of any shape

    Returns:
    A -- output of sigmoid(Z), same shape as Z
    cache -- returns Z as well, useful during backpropagation
    """
    A = 1. / (1. + np.exp(-Z))
    cache = Z

    return A, cache


def sigmoid_grad(dA, cache):
    """
    Implement the backward propagation for a single SIGMOID unit.

    Arguments:
    dA    -- post-activation gradient, of any shape
    cache -- 'Z' where we store for computing backward propagation efficiently

    Returns:
    dZ    -- Gradient of the cost with respect to Z
    """
    Z = cache
    s, cache_sigmoid = sigmoid(Z)
    dZ = dA * s * (1-s)

    assert (dZ.shape == Z.shape)

    return dZ


def test_sigmoid_and_its_gradient():
    print("Running basic tests...")
    x = np.array([[1, 2], [-1, -2]])
    f, _ = sigmoid(x)
    g = sigmoid_grad(1,x)

    print(f)
    f_ans = np.array([
        [0.73105858, 0.88079708],
        [0.26894142, 0.11920292]])
    assert np.allclose(f, f_ans, rtol=1e-05, atol=1e-06)

    print(g)
    g_ans = np.array([
        [0.19661193, 0.10499359],
        [0.19661193, 0.10499359]])
    assert np.allclose(g, g_ans, rtol=1e-05, atol=1e-06)


if __name__ == "__main__":
    test_sigmoid_and_its_gradient()
