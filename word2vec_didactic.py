#!/usr/bin/env python

import numpy as np
import random
import sys
sys.path.insert(0, "./activations")
sys.path.insert(1, "./utils")
from softmax import *
from dnn import *
from gradient_check import *

"""
This is a didactic implementation of Word2Vec just to reuse the general purpose feed-forward
network available in dnn.py and to understand each step of the forward and the back propagation.
Only the skip-gram model is currently implemented.
For the more efficiency you can use the word2vec.py script.
"""

def normalize_rows(X):
    """
    Row normalization function. Implement a function that normalizes each
    row of a matrix to have unit length

    Arguments:
    X -- data, numpy array of shape (number of examples, input size)

    Returns:
    X -- normalized data, numpy array of shape (number of examples, input size)
    """
    # Normalization according to rows (axis = 1)
    Y = np.linalg.norm(X, axis=1, keepdims=True)
    X /= Y

    return X


def word2vec_forward(X, parameters, hyper_parameters):
    # For more details on the forward propagation implementation see dnn.py
    A, cache = forward_propagation(X, parameters, hyper_parameters)

    return A, cache


def word2vec_backward(AL, Y, caches, hyper_parameters):
    # For more details on the backward propagation implementation see dnn.py
    gradients = backpropagation(AL, Y, caches, hyper_parameters)

    return gradients


def test_w2v():
    random.seed(31415)
    np.random.seed(9265)

    dataset = type('dummy', (), {})() # It creates class dynamically and creates an instance of it
    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    # Set parameters and hyper_parameters
    parameters = {}
    hyper_parameters = {}

    # This a simple vocabulary: each value represents the position of the 1 element in the one-hot-vector
    # For instance 'a' will be [1,0,0,0,0]
    vocabulary = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    # Generate parameters (or weights) of the hidden and of the output layer in one single matrix that must be splitted in W0 and W1
    # In this specific case 5 is the dimension of the vocabulary (5*2 = 10) and 3 is the dimension of the embeddings for the words
    weights = normalize_rows(np.random.randn(10,3))

    # Number of different words in the vocabulary
    m = weights.shape[0]

    # Parameters initizialization based on the random generated weights
    parameters["W1"] = weights[:int(m/2),:].T
    parameters["b1"] = np.zeros((parameters["W1"].shape[0],1))
    parameters["W2"] = weights[int(m/2):,]
    parameters["b2"] = np.zeros((parameters["W2"].shape[0],1))
    activations = {}
    activations[1] = "linear"
    activations[2] = "softmax"
    hyper_parameters["activations"] = activations

    # Randomize center word and context word generation
    C = 5
    C1 = random.randint(1,C)
    centerword, context = dataset.getRandomContext(C1) # Example of output: ('c', ['a', 'b', 'e'])

    # Initialize the one-hot vector representing the center word (input data in the Skip-gram model)
    X = np.zeros((len(vocabulary), 1))
    X[vocabulary[centerword]] = 1

    # Initialize the one-hot vector representing the first word of the context
    Y = np.zeros((len(vocabulary), 1))
    Y[vocabulary[context[0]]] = 1

    print("\nVocabulary: " + str(vocabulary))
    print("\nCenter word: " + str(centerword))
    print("\nCenter word represented as one-hot-vector: \n\n" + str(X))
    print("\nShape of input data: " + str(X.shape))
    print("\nContext: " + str(context))
    print("\nFirst word of the context: " + str(context[0]))
    print("\nFirst word of the context represented as one-hot-vector: \n\n" + str(Y))
    print("\nHidden parameters (transposed): \n\n" + str(parameters["W1"]))
    print("\nShape of hidden parameters: " + str(parameters["W1"].shape))
    print("\nOutput parameters: \n\n" + str(parameters["W2"]))

    AL, caches = word2vec_forward(X, parameters, hyper_parameters)

    print("\nCache values for the back propagation: ")
    print("\n* Linear cache: ")
    print("\n** Cache of the first layer: ")
    print("\n*** Input of the first layer: \n\n" + str(caches[0][0][0]))
    print("\n*** W1 (parameters of the first layer): \n\n" + str(caches[0][0][1]))
    print("\n*** b1 (parameters of the first layer): \n\n" + str(caches[0][0][2]))
    print("\n*** Output of the first layer (linear activation): \n\n" + str(caches[0][1]))
    print("\n** Cache of the second layer: ")
    print("\n*** Input of the second layer: \n\n" + str(caches[1][0][0]))
    print("\n*** W2 (parameters of the second layer): \n\n" + str(caches[1][0][1]))
    print("\n*** b2 (parameters of the second layer): \n\n" + str(caches[1][0][2]))
    print("\n*** Output of the second layer (linear activation): \n\n" + str(caches[1][1]))

    # Test to verify that I have correctly extracted the cache values
    A_from_cache, _ = softmax(caches[1][1])
    assert np.allclose(AL, A_from_cache, rtol=1e-05, atol=1e-06)

    print("\nOutput of the forward propagation: \n\n" + str(AL))

    # Compute cross-entropy loss function
    loss = compute_loss(AL, Y)
    print("\nThe value of loss is: " + str(loss))

    # Backward propagation
    gradients = word2vec_backward(AL, Y, caches, hyper_parameters)
    print("\n* Gradients: ")
    print("\n** Gradients of the output layer: ")
    print("\n*** Gradients of W2: \n\n" + str(gradients["dW2"]))
    print("\n*** Gradients of W1: \n\n" + str(gradients["dW1"]))

    print("\nParameters update using gradient descent...")

    new_parameters = update_parameters(parameters, gradients, 0.1)

    print(new_parameters)

    print("\nAnother forward propagation step with new parameters...")

    AL, caches = word2vec_forward(X, new_parameters, hyper_parameters)

    new_loss = compute_loss(AL, Y)

    print("\nThe new value of loss is: " + str(new_loss))

    print("\nCompare loss after one step of gradient descent: " + str(new_loss) + " < " + str(loss))

    print("\nThank God!")

def dummySampleTokenIdx():
    # It generates randomly an int between 0 and 4
    return random.randint(0, 4)


def getRandomContext(C):
    # C is equal to the number of elements in the context (window)
    # Example of output: ('b', ['c', 'a']) if C is 1
    # Example of output: ('c', ['c', 'b', 'e', 'a', 'b', 'e']) if C is 3
    tokens = ["a", "b", "c", "d", "e"]
    return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in range(2*C)] # C is a window


if __name__ == "__main__":
    test_w2v()
    print("\n"  + "Launch" + "\033[92m" + " python tests/word2vec_test.py " + "\033[0m" + "script to test advanced versions of Word2Vec cost and gradient\n")
