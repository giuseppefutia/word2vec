#!/usr/bin/env python

import numpy as np
import sys
sys.path.insert(0, './activations')
from dnn import *
from sigmoid import *
from relu import *


def dictionary_to_vector(parameters):
    """
    Roll all our parameters dictionary into a single vector
    """
    keys = []
    count = 0
    for key in ["W1", "b1", "W2", "b2", "W3", "b3"]:

        # flatten parameter
        new_vector = np.reshape(parameters[key], (-1,1))
        keys = keys + [key]*new_vector.shape[0]

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta, keys


def vector_to_dictionary(theta):
    """
    Unroll all our parameters dictionary from a single vector
    """
    parameters = {}
    parameters["W1"] = theta[:20].reshape((5,4))
    parameters["b1"] = theta[20:25].reshape((5,1))
    parameters["W2"] = theta[25:40].reshape((3,5))
    parameters["b2"] = theta[40:43].reshape((3,1))
    parameters["W3"] = theta[43:46].reshape((1,3))
    parameters["b3"] = theta[46:47].reshape((1,1))

    return parameters


def gradients_to_vector(gradients):
    """
    Roll all our gradients dictionary into a single vector
    """

    count = 0
    for key in ["dW1", "db1", "dW2", "db2", "dW3", "db3"]:
        # flatten parameter
        new_vector = np.reshape(gradients[key], (-1,1))

        if count == 0:
            theta = new_vector
        else:
            theta = np.concatenate((theta, new_vector), axis=0)
        count = count + 1

    return theta


def gradient_check(parameters, hyper_parameters, gradients, X, Y, epsilon = 1e-7):
    """
    Checks if backward_propagation_n computes correctly the gradient of the cost output by forward_propagation

    Arguments:
    parameters -- python dictionary containing your parameters "W1", "b1", "W2", "b2", "W3", "b3":
    hyper_parameters -- python dictionary containing hyper parameters like activation functions (see utils.py)
    grad -- output of backward_propagation_n, contains gradients of the cost with respect to the parameters.
    X -- input datapoint, of shape (input size, 1)
    Y -- true "label"
    epsilon -- tiny shift to the input to compute approximated gradient with formula(1)

    Returns:
    difference -- difference (2) between the approximated gradient and the backward propagation gradient
    """

    # Set-up variables
    parameters_values, _ = dictionary_to_vector(parameters)
    grad = gradients_to_vector(gradients)
    num_parameters = parameters_values.shape[0]
    J_plus = np.zeros((num_parameters, 1))
    J_minus = np.zeros((num_parameters, 1))
    gradapprox = np.zeros((num_parameters, 1))

    # Compute gradapprox
    for i in range(num_parameters):

        # Compute J_plus[i]. Inputs: "parameters_values, epsilon". Output = "J_plus[i]"
        thetaplus = np.copy(parameters_values)
        thetaplus[i][0] =  thetaplus[i][0] + epsilon
        AL_plus, _ = forward_propagation(X, vector_to_dictionary(thetaplus), hyper_parameters)
        J_plus[i] = compute_cost(AL_plus, Y)

        # Compute J_minus[i]. Inputs: "parameters_values, epsilon". Output = "J_minus[i]"
        thetaminus = np.copy(parameters_values)
        thetaminus[i][0] = thetaminus[i][0] - epsilon
        AL_minus, _ = forward_propagation(X, vector_to_dictionary(thetaminus), hyper_parameters)
        J_minus[i] = compute_cost(AL_minus, Y)

        # Compute gradapprox[i]
        gradapprox[i] = (J_plus[i] - J_minus[i]) / (2 * epsilon)

    # Compare gradapprox to backward propagation gradients by computing difference.
    numerator = np.linalg.norm(grad-gradapprox)
    denominator = np.linalg.norm(grad+gradapprox)
    difference = numerator / denominator

    if difference > 2e-7:
        print ("\033[93m" + "There is a mistake in the backward propagation! difference = " + str(difference) + "\033[0m")
    else:
        print ("\033[92m" + "Your backward propagation works perfectly fine! difference = " + str(difference) + "\033[0m")

    return difference

def test_gradient_check():
    np.random.seed(1)
    X = np.random.randn(4,3)
    Y = np.array([1, 1, 0])
    W1 = np.random.randn(5,4)
    b1 = np.random.randn(5,1)
    W2 = np.random.randn(3,5)
    b2 = np.random.randn(3,1)
    W3 = np.random.randn(1,3)
    b3 = np.random.randn(1,1)
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2,
                  "W3": W3,
                  "b3": b3}
    hyper_parameters = {}
    hyper_parameters["activations"] = {}
    hyper_parameters["activations"][1] = "relu"
    hyper_parameters["activations"][2] = "relu"
    hyper_parameters["activations"][3] = "sigmoid"

    AL, caches = forward_propagation(X, parameters, hyper_parameters)
    cost = compute_cost(AL, Y)
    gradients = backpropagation(AL, Y, caches)
    difference = gradient_check(parameters, hyper_parameters, gradients, X, Y)


if __name__ == "__main__":
    test_gradient_check()
