#!/usr/bin/env python

import numpy as np


def initialize_hyper_parameters(layer_acts, learning_rate):
    """
    Initialize parameters for different levels of the network

    Arguments:
    layer_acts -- python array (list) containing the activation functions of each layer in the network
    learning_rate -- float value used as constant for gradient descent

    Returns:
    hyper_parameters -- python dictionary containing hyper_parameters (can be further extended)

    """
    hyper_parameters = {}
    activations = {}
    L = len(layer_acts) # number of layers in the network
    for l in range(0, L):
        activations[l+1] = layer_acts[l]
    hyper_parameters["activations"] = activations
    hyper_parameters["learning_rate"] = learning_rate

    return hyper_parameters


def test_initialize_hyper_parameters():
    print("\033[92m" + "\nTest initialize_hyper_parameters() ..." + "\033[0m")
    layer_acts = ["relu", "relu", "sigmoid"]
    learning_rate = 0.0075
    hyper_parameters = initialize_hyper_parameters(layer_acts, learning_rate)
    print(hyper_parameters["activations"])

    assert len(hyper_parameters["activations"]) == 3
    assert hyper_parameters["activations"][1] == "relu"

    print("\033[92m" + "... end test" + "\033[0m")


def initialize_parameters(layer_dims):
    """
    Initialize parameters for different levels of the network

    Arguments:
    layer_dims -- python array (list) containing the dimensions of each layer in the network

    Returns:
    parameters -- python dictionary containing your parameters "W1", "b1", ..., "WL", "bL", ...:
                    Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
                    bl -- bias vector of shape (layer_dims[l], 1)
    """

    np.random.seed(1)
    parameters = {}
    L = len(layer_dims) # number of layers in the network

    for l in range(1, L):
        parameters['W' + str(l)] = np.random.randn(layer_dims[l], layer_dims[l-1]) * 0.01
        parameters['b' + str(l)] = np.zeros((layer_dims[l], 1))
        assert(parameters['W' + str(l)].shape == (layer_dims[l], layer_dims[l-1]))
        assert(parameters['b' + str(l)].shape == (layer_dims[l], 1))

    return parameters


def test_initialize_parameters():
    print("\n" + "\033[92m" + "Test initialize_parameters() ..." + "\033[0m")

    np.random.seed(1)
    parameters = initialize_parameters([3,2,1])

    print("W1 = " + str(parameters["W1"]))
    print("b1 = " + str(parameters["b1"]))
    print("W2 = " + str(parameters["W2"]))
    print("b2 = " + str(parameters["b2"]))

    W1 = parameters["W1"]
    W1_expected = np.array([[0.01624345,-0.00611756,-0.00528172],[-0.01072969,0.00865408,-0.02301539]])
    assert np.allclose(W1, W1_expected, rtol=1e-05, atol=1e-06)

    b1 = parameters["b1"]
    b1_expected = np.array([[0.],[0.]])
    assert np.allclose(b1, b1_expected, rtol=1e-05, atol=1e-06)

    W2 = parameters["W2"]
    W2_expected = np.array([[0.01744812, -0.00761207]])
    assert np.allclose(W2, W2_expected, rtol=1e-05, atol=1e-06)

    b2 = parameters["b2"]
    b2_expected = np.array([[ 0.]])
    assert np.allclose(b2, b2_expected, rtol=1e-05, atol=1e-06)

    print("\033[92m" + "... end test" + "\033[0m")


if __name__ == "__main__":
    test_initialize_hyper_parameters()
    test_initialize_parameters()
