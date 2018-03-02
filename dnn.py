#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
import sys
sys.path.insert(0, "./activations")
sys.path.insert(1, "./utils")
from relu import *
from sigmoid import *
from softmax import *

"""
TODO: check if softmax can be included in linear activation or it must be directly combined with cross entropy
TODO: Use the general cross-entropy to compute the cost
TODO: Update forward and back propagation using hyper parameters for activation functions
TODO: Reduce backpropagation(), considering all layers in the same way
"""

def linear_forward(A, W, b):
    """
    Implement the linear part of a layer's forward propagation.

    Arguments:
    A -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """

    Z = np.dot(W, A) + b

    assert(Z.shape == (W.shape[0], A.shape[1]))
    cache = (A, W, b)

    return Z, cache


def linear_activation_forward(A_prev, W, b, activation):
    """
    Implement the forward propagation for the LINEAR->ACTIVATION layer

    Arguments:
    A_prev -- activations from previous layer (or input data): (size of previous layer, number of examples)
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (size of the current layer, 1)
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    A -- the output of the activation function, also called the post-activation value
    cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
    """

    if activation == "sigmoid":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = sigmoid(Z)

    if activation == "relu":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = relu(Z)

    if activation == "softmax":
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A, activation_cache = softmax(Z)

    if activation == "linear":
        # A particular case in which there is no activation function (useful for Word2Vec)
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = linear_forward(A_prev, W, b)
        A = Z
        activation_cache = Z

    assert (A.shape == (W.shape[0], A_prev.shape[1]))
    cache = (linear_cache, activation_cache)

    return A, cache


def forward_propagation(X, parameters, hyper_parameters):
    """
    Forward propagation algorithm
    It extends forward propagation for the [LINEAR->RELU]*(L-1)->LINEAR->SIGMOID computation written by Andrew Ng

    Arguments:
    X -- data, numpy array of shape (input size, number of examples)
    parameters -- output of initialize_parameters() function in utils
    hyper_parameters -- output of initialize_hyper_parameters() function in utils

    Returns:
    AL -- last post-activation value
    caches -- list of caches containing every cache of linear activation forward
    """

    caches = []
    A = X
    L = len(parameters) // 2 # number of layers in the neural network.

    # Implement [LINEAR -> ACTIVATION]
    for l in range(1, L+1):
        A_prev = A
        A, cache = linear_activation_forward(A_prev,
                                             parameters["W" +  str(l)],
                                             parameters["b" +  str(l)],
                                             hyper_parameters["activations"][l])
        caches.append(cache)

    # assert(A.shape == (1, X.shape[1])) TODO: Check if this control is correct in any case

    return A, caches


def compute_loss(AL, Y):
    """
    Compute the cross-entropy loss.

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    loss -- cross-entropy loss
    """
    # In case of loss of logistic regression you can compute np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
    # but I prefer to generalize the loss function for multiclass problems.
    # Cross entropy indicates the distance between what the model believes the output distribution should be, and what the original distribution really is
    loss = - Y * np.log(AL)
    loss = np.squeeze(np.sum(loss))

    return loss


def compute_cost(AL, Y):
    """
    Compute the average of the loss contribution for each sample

    Arguments:
    AL -- probability vector corresponding to your label predictions, shape (1, number of examples)
    Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (1, number of examples)

    Returns:
    cost -- cross-entropy cost
    """
    m = AL.shape[1]
    cost = (1. / m) * np.sum(compute_loss(AL, Y))
    cost = np.squeeze(cost) # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).

    return cost


def linear_backward(dZ, cache):
    """
    Implement the linear portion of backward propagation for a single layer (layer l)

    Arguments:
    dZ -- Gradient of the cost with respect to the linear output (of current layer l)
    cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = (1/m) * np.dot(dZ, A_prev.T)
    db = (1/m) * np.sum(dZ, axis=1, keepdims=True)
    dA_prev = np.dot(W.T, dZ)

    assert (dA_prev.shape == A_prev.shape)
    assert (dW.shape == W.shape)
    assert (db.shape == b.shape)

    return dA_prev, dW, db


def linear_activation_backward(dA, cache, activation):
    """
    Implement the backward propagation for the LINEAR->ACTIVATION layer.

    Arguments:
    dA -- post-activation gradient for current layer l
    cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
    activation -- the activation to be used in this layer, stored as a text string: "sigmoid" or "relu"

    Returns:
    dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
    dW -- Gradient of the cost with respect to W (current layer l), same shape as W
    db -- Gradient of the cost with respect to b (current layer l), same shape as b
    """

    linear_cache, activation_cache = cache

    if activation == "relu":
        dZ = relu_grad(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "sigmoid":
        dZ = sigmoid_grad(dA, activation_cache)
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    elif activation == "softmax":
        dZ = softmax_grad(dA, activation_cache)
        rows, cols = np.nonzero(dZ)
        dZ_reshaped = dZ[rows, cols].reshape([dZ.shape[0], 1])
        dA_prev, dW, db = linear_backward(dZ_reshaped, linear_cache)

    elif activation == "linear":
        dZ = dA
        dA_prev, dW, db = linear_backward(dZ, linear_cache)

    return dA_prev, dW, db


def backpropagation(AL, Y, caches, hyper_parameters):
    """
    Implement the backward propagation

    Arguments:
    AL -- probability vector, output of the forward propagation
    Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
    caches -- list of caches containing every cache of linear_activation_forward() with the activation layer
    hyper_parameters -- hyper parameters of the networks (in this case I need activation functions)


    Returns:
    grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ...
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ...
    """
    grads = {}
    L = len(caches) # the number of layers
    m = AL.shape[1] # the number of samples at the output layer
    Y = Y.reshape(AL.shape) # after this line, Y has the same shape of AL

    # The first step of the backpropagation changes according to the activation function of the last layer
    if (hyper_parameters["activations"][L] == "sigmoid"):
        # Compute the derivative of the cross entropy cost for logistic regression:
        # np.multiply(-np.log(AL), Y) + np.multiply(-np.log(1 - AL), 1 - Y)
        dAL = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))

    elif(hyper_parameters["activations"][L] == "softmax"):
        # Compute the derivative of the cross entropy cost for a multilabel classifier.
        # Y * np.log(AL)
        # You obtain a vector like [0,0,0,1/AL,0] because all elements of vector Y are 0s except 1
        dAL = - np.divide(Y, AL)

    current_cache = caches[-1]
    grads["dA" + str(L)], grads["dW" + str(L)], grads["db" + str(L)] = linear_activation_backward(dAL, current_cache, hyper_parameters["activations"][L])

    for l in reversed(range(L-1)):
        current_cache = caches[l]
        dA_prev_temp, dW_temp, db_temp = linear_activation_backward(grads["dA" + str(l + 2)], current_cache, hyper_parameters["activations"][l+1])
        grads["dA" + str(l + 1)] = dA_prev_temp
        grads["dW" + str(l + 1)] = dW_temp
        grads["db" + str(l + 1)] = db_temp

    return grads


def update_parameters(parameters, grads, learning_rate):
    """
    Update parameters using gradient descent

    Arguments:
    parameters -- python dictionary containing  parameters
    grads -- python dictionary containing gradients, output of backpropagation

    Returns:
    parameters -- python dictionary containing your updated parameters
                  parameters["W" + str(l)] = ...
                  parameters["b" + str(l)] = ...
    """

    L = len(parameters) // 2 # number of layers in the neural network

    # Update rule for each parameter. Use a for loop.
    for l in range(L):
        parameters["W" + str(l + 1)] = parameters["W" + str(l + 1)] - learning_rate * grads["dW" + str(l + 1)]
        parameters["b" + str(l + 1)] = parameters["b" + str(l + 1)] - learning_rate * grads["db" + str(l + 1)]

    return parameters


def predict(X, y, parameters, hyper_parameters):
    """
    This function is used to predict the results of a  L-layer neural network.

    Arguments:
    X -- data set of examples you would like to label
    parameters -- parameters of the trained model
    hyper_parameters -- hyper parameters of the networks (in this case I need activation functions for the forward propagation)

    Returns:
    p -- predictions for the given dataset X
    """

    m = X.shape[1]
    n = len(parameters) // 2 # number of layers in the neural network
    p = np.zeros((1, m),dtype=int)

    # Forward propagation
    probas, caches = forward_propagation(X, parameters, hyper_parameters)

    # convert probas to 0/1 predictions
    for i in range(0, probas.shape[1]):
        if probas[0,i] > 0.5:
            p[0,i] = 1
        else:
            p[0,i] = 0

    #print results
    #print ("predictions: " + str(p))
    #print ("true labels: " + str(y))
    print("Accuracy: %s" % str(np.sum(p == y)/float(m)))

    return p

if __name__ == "__main__":
    print("\n"  + "Launch" + "\033[92m" + " python tests/dnn_test.py " + "\033[0m" + "script to test the Neural Network\n")
