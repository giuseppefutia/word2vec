#!/usr/bin/env python

import numpy as np
import sys
sys.path.insert(0, "./activations")
from dnn import *
from softmax import *
from gradient_check_naive import *

"""
TODO: check if you have to normalize columns instead of rows
TODO: If linear_activation_forward() take as input the hyper_parameters python array,
      you should update softmax_cost_and_gradient()
TODO: update softmax_cost_and_gradient() using functions in dnn.py
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


def test_normalize_rows():
    print("\n" + "\033[92m" + "Test normalize_rows() ..." + "\033[0m")
    x = normalize_rows(np.array([[3.0,4.0],[1, 2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("\033[92m" + "... end test" + "\033[0m")


def softmax_cost_and_gradient(predicted, target, output_vectors, dataset):
    """ Cost and gradients for one predicted word vector and one target
    word vector as a building block for word2vec models, assuming the
    softmax prediction function and cross entropy loss.

    Arguments:
    predicted -- numpy (m,n), output of the forward_propagation()
    target -- int, the index of the target word
    output_vectors -- "output" vectors (as rows) for all tokens (words in the vocabulary)
    dataset -- needed for negative sampling, unused here.

    Return:
    cost -- cross entropy cost for the softmax word prediction
    grad_pred -- the gradient with respect to the predicted word vector
    grad -- the gradient with respect to all the other word vectors
    """

    # In output_vectors all the words of my dictionary are represented as rows
    # For this reason, the shape of output_vectors is (number of words, word vector)
    m = output_vectors.shape[0]
    n = output_vectors.shape[1]

    # It represents the one-hot vector as output:
    # so it should be long as number or words (vocabulary size)
    Y = np.zeros(m)

    # The position target of the one-hot vector is initialize to one
    Y[target] = 1

    W = predicted
    A = output_vectors.T
    b = 0

    # Forward propagation (TODO: maybe you have to substitute it)
    probabilities, _ = linear_activation_forward(A, W, b, "softmax")

    # Cross entropy cost
    cost = np.sum(-Y * np.log(probabilities))

    # Backward propagation (TODO: maybe you have to substitute it)
    dout = probabilities - Y # (1, n_words)

    # TODO: Understand what are these two gradients
    grad_pred = np.dot(dout, output_vectors) # (1, dim_embed)

    grad = np.dot(dout.T, predicted) # (n_words, dim_embed)

    return cost, grad_pred, grad


def skipgram(current_word, C, context_words, tokens, input_vectors, output_vectors,
             dataset, word2vec_cost_and_gradient=softmax_cost_and_gradient):
    """ Skip-gram model in word2vec

    Arguments:
    current_word -- a string of the current center word
    C -- integer, context size
    context_words -- list of no more than 2*C strings, the context words
    tokens -- a dictionary that maps words to their indices in
              the word vector list
    dataset -- object that defines current_word and context_words
    input_vectors -- "input" word vectors (as rows) for all tokens
    output_vectors -- "output" word vectors (as rows) for all tokens
    word2vec_cost_and_gradient -- the cost and gradient function for
                               a prediction vector given the target
                               word vectors, could be one of the two
                               cost functions you implemented above.

    Return:
    cost -- the cost function value for the skip-gram model
    grad -- the gradient with respect to the word vectors
    """

    cost = 0.0
    gradIn = np.zeros(input_vectors.shape)
    gradOut = np.zeros(output_vectors.shape)

    idx = tokens[current_word] # tokens['a'] = 1
    input_vector = input_vectors[idx:idx+1] # (1, dim_embed)

    for context in context_words:
        c, g_in, g_out = word2vec_cost_and_gradient(input_vector, tokens[context], output_vectors, dataset)
        cost += c
        gradIn[idx:idx+1, :] += g_in
        gradOut += g_out

    return cost, gradIn, gradOut


#############################################
# Testing functions
#############################################

def word2vec_sgd_wrapper(word2vecModel, tokens, wordVectors, dataset, C,
                         word2vecCostAndGradient=softmax_cost_and_gradient):
    batchsize = 50
    cost = 0.0
    grad = np.zeros(wordVectors.shape)
    N = wordVectors.shape[0]
    inputVectors = wordVectors[:int(N/2),:]
    outputVectors = wordVectors[int(N/2):,:]

    for i in range(batchsize):
        C1 = random.randint(1,C)
        centerword, context = dataset.getRandomContext(C1)

        if word2vecModel == skipgram:
            denom = 1
        else:
            denom = 1

        c, gin, gout = word2vecModel(
            centerword, C1, context, tokens, inputVectors, outputVectors,
            dataset, word2vecCostAndGradient)
        cost += c / batchsize / denom
        grad[:int(N/2), :] += gin / batchsize / denom
        grad[int(N/2):, :] += gout / batchsize / denom

    return cost, grad


def test_word2vec():
    """
    Test Word2Vec models

    """

    dataset = type('dummy', (), {})() # It creates class dynamically and creates an instance of it

    def dummySampleTokenIdx(): # It generates randomly an int between 0 and 4
        return random.randint(0, 4)

    # Example of output: ('b', ['c', 'a'])
    # Example of output: ('c', ['c', 'b', 'e', 'a', 'b', 'e'])
    def getRandomContext(C): # It generates an example of context
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in range(2*C)] # C is a window

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)
    dummy_vectors = normalize_rows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("\n==== Gradient check for skip-gram ====")
    gradcheck_naive(lambda vec: word2vec_sgd_wrapper(
                                    skipgram,
                                    dummy_tokens,
                                    vec,
                                    dataset,
                                    5,
                                    softmax_cost_and_gradient),
                                    dummy_vectors)


if __name__ == "__main__":
    test_normalize_rows()
    test_word2vec()
