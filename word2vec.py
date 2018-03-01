#!/usr/bin/env python

import numpy as np
import random
import sys
sys.path.insert(0, "./activations")
from softmax import *
from dnn import *

"""
TODO: Update backpropagation with the implementation in dnn.py
TODO: Extend the use of the forward propagation implementation in dnn.py also for negative sampling
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


def softmax_cost_grads(input_vector, output_vectors, target_index, dataset):
    """ Cost and gradients for one predicted word vector and one target
    word vector as a building block for word2vec models, assuming the
    softmax prediction function and cross entropy loss.

    Arguments:
    input_vector -- matrix (1,n), representation of the center word in the input matrix
    output_vectors -- "output" vectors (as columns) for all tokens (words in the vocabulary)
    target_index -- index of the target word
    dataset -- object with sample data useful for the negative sampling (not used here)

    Return:
    cost -- cross-entropy cost
    grad_pred -- the gradient with respect to the predicted word vector
    grad -- the gradient with respect to all the other word vectors
    """

    # Forward propagation (consider only the last layer for efficiency reasons)
    # See implementation of linear_activation_forward(A_prev, W, b, activation) in dnn.py
    A_prev = input_vector.T
    W = output_vectors
    b = np.zeros((W.shape[0],1))
    activation = "softmax"
    probabilities, cache = linear_activation_forward(A_prev, W, b, activation)

    # Cost value
    cost = -np.log(probabilities[target_index])

    # Backward propagation

    # First step
    probabilities[target_index] -= 1 # (n_words, 1)
    delta_out = probabilities
    delta_out = delta_out.reshape(probabilities.shape[0]) # (n_words,)

    # Second step (gradients of weights of the second layer)
    grad_pred = np.dot(delta_out, output_vectors) # (1, dim_embed)

    # Third step (gradients of weights of the first layer)
    # See the implementation of the linear_backward(dZ, cache) in dnn.py
    _, grad, _ = linear_backward(delta_out.reshape(delta_out.shape[0], 1), cache[0]) # (n_words, dim_embed)

    return cost, grad_pred, grad


def softmax_cost_grads_reg(features, labels, weights, regularization = 0.0, nopredictions = False):
    """
    Softmax cost and gradient with regularization

    Arguments:
    features -- feature vectors, each row is a feature vector
    labels -- labels corresponding to the feature vectors
    weights -- weights of the regressor
    regularization -- L2 regularization constant

    Output:
    cost -- cost of the regressor
    grad -- gradient of the regressor cost with respect to its weights
    pred -- label predictions of the regressor

    """
    probabilities, _ = softmax(features.dot(weights).T)

    if len(features.shape) > 1:
        N = features.shape[0]
    else:
        N = 1

    # A vectorized implementation of 1/N * sum(cross_entropy(x_i, y_i)) + 1/2*|w|^2

    cost = np.sum(-np.log(probabilities[labels, range(N)])) / N
    cost += 0.5 * regularization * np.sum(weights ** 2)

    grad = np.zeros_like(weights)
    pred = 0;

    numlabels = np.shape(weights)[1]

    delta = probabilities.T - np.eye(numlabels)[labels]
    grad = (np.dot(features.T,delta) / N) + regularization*weights

    if N > 1:
        pred = np.argmax(probabilities, axis=0)
    else:
        pred = np.argmax(probabilities)

    if nopredictions:
        return cost, grad
    else:
        return cost, grad, pred


def softmax_wrapper(features, labels, weights, regularization = 0.0):
    cost, grad, _ = softmax_cost_grads_reg(features, labels, weights, regularization)
    return cost, grad


def negative_sampling(input_vector, output_vectors, target_index, dataset, K=10):
    """ Negative sampling cost and gradients function for Word2Vec models

    Arguments:
    input_vector -- matrix (1,n), representation of the center word in the input matrix
    output_vectors -- "output" vectors (as columns) for all tokens (words in the vocabulary)
    target_index -- index of the target word
    dataset -- object with sample data useful for the negative sampling

    Returns:
    cost -- cross-entropy cost
    grad_pred -- the gradient with respect to the predicted word vector (hidden layer)
    grad -- the gradient with respect to all the other word vectors (output layer)

    """

    grad_pred = np.zeros_like(input_vector)
    grad = np.zeros_like(output_vectors)

    indices = [target_index]

    # Generate K int numbers for negative sampling
    for k in range(K):
        newidx = dataset.sampleTokenIdx()
        while newidx == target_index:
            newidx = dataset.sampleTokenIdx()
        indices += [newidx]

    directions = np.array([1] + [-1 for k in range(K)])

    N = np.shape(output_vectors)[1]

    output_words = output_vectors[indices,:]
    input_vector = input_vector.reshape(input_vector.shape[1])

    delta, _ = sigmoid(np.dot(output_words, input_vector) * directions)
    delta_minus = (delta - 1) * directions;

    # cost function in the case of sigmoid
    cost = -np.sum(np.log(delta));

    grad_pred = np.dot(delta_minus.reshape(1,K+1), output_words).flatten()
    grad_min = np.dot(delta_minus.reshape(K+1,1), input_vector.reshape(1,N))

    for k in range(K+1):
        grad[indices[k]] += grad_min[k,:]

    return cost, grad_pred, grad


def skipgram(current_word, C, context_words, tokens, input_vectors,
             output_vectors, dataset, word2vec_cost_grads=softmax_cost_grads):
    """ Skip-gram model in Word2Vec

    Arguments:
    current_word -- a string of the current center word
    C -- integer, context size
    context_words -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in the word vector list
    input_vectors -- word vectors at the hidden layer
    output_vectors -- word vectors at the output layer
    dataset -- object that defines current_word and context_words
    word2vec_cost_grads -- cost and gradient function for a prediction vector given the target word vectors

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """
    cost = 0.0
    grad_in = np.zeros(input_vectors.shape) # IMPORTANT: Remember that you have transposed this matrix
    grad_out = np.zeros(output_vectors.shape)

    # tokens['a'] = 0 --> It should represent one-hot vector of this type [1,0,0,0,0...,0]
    idx = tokens[current_word]

    # Compute the cost and the gradient for each context word
    for context in context_words:

        input_vector = input_vectors[:,idx]
        input_vector = input_vector.reshape(1, input_vector.shape[0])

        # Cost and Gradients could be caluculated with the softmax_cost_grads, the softmax_cost_grads_reg or the negative_sampling
        dcost, g_in, g_out = word2vec_cost_grads(input_vector,
                                                 output_vectors,
                                                 tokens[context],
                                                 dataset)

        cost += dcost
        grad_in[:,idx] += g_in
        grad_out += g_out

    return cost, grad_in, grad_out


def word2vec_sgd_wrapper(word2vec_model, tokens, word_vectors, dataset, C,
                         word2vec_gradient=softmax_cost_grads):
    """
    Wrap the Word2Vec model in order to run in batch

    Arguments:
    word2vec_model -- model of Word2Vec (currently only Skip-gram is available)
    tokens -- tokens of all words in the vocabulary
    word_vectors -- embeddings (or weights) to represent words to code and decode the vocabulary
    dataset -- structure that contains, for each sentence, the center word and the context
    C -- dimension of the window to get the context
    word2vec_gradient -- function to compute the cost and the gradient (default: softmax_cost_grads)

    Returns:
    cost -- Cost generated by the training stage
    grad -- Gradients computed during the training stage
    """
    # It defines number of samples that going to be propagated through the network.
    # It means that each 50 samples you update your parameters (efficient reasons)
    batchsize = 50
    cost = 0.0
    grad = np.zeros(word_vectors.shape) # (m,n) Zero matrix for the gradients
    m = word_vectors.shape[0] # Number of different words in the vocabulary

    # Matrices of parameters for the forward propagation
    input_vectors = word_vectors[:int(m/2),:].T
    output_vectors = word_vectors[int(m/2):,]

    for i in range(batchsize):

        # Randomize center word and context word generation
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1) # Example of output: ('c', ['a', 'b', 'e'])

        # Cost and Gradients could be caluculated with the softmax_cost_grads, the softmax_cost_grads_reg or the negative_sampling
        c, gin, gout = word2vec_model(centerword, C1, context, tokens,
                                      input_vectors, output_vectors, dataset,
                                      word2vec_gradient)

        # In the mini-batch approach, you divide all things for the batchsize
        cost += c / batchsize

        grad[:int(m/2),:] += gin.T / batchsize
        grad[int(m/2):,] += gout / batchsize

    # Return cost and grad every batchsize
    return cost, grad


if __name__ == "__main__":
    print("\n"  + "Launch" + "\033[92m" + " python tests/word2vec_test.py " + "\033[0m" + "script to test Word2Vec cost and gradient\n")
