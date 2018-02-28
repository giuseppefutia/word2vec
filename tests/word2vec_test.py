#!/usr/bin/env python

import sys
sys.path.insert(0, "./")
from word2vec import *
from gradient_check_naive import *
sys.path.insert(1, "./applications/sentiment")
from stanford import *
from sent_utils import *
from stochastic_gradient_descent import *


def test_normalize_rows():
    print("\n" + "\033[92m" + "Test normalize_rows() ..." + "\033[0m")
    x = normalize_rows(np.array([[3.0,4.0],[1,2]]))
    print(x)
    ans = np.array([[0.6,0.8],[0.4472136,0.89442719]])
    assert np.allclose(x, ans, rtol=1e-05, atol=1e-06)
    print("\033[92m" + "... end test" + "\033[0m")


def test_negative_sampling():
    print("\n" + "\033[92m" + "Test negative_sampling() instructions..." + "\033[0m")
    # Recall some scripts belonging to the negative sampling function to verify the correct behaviour
    # of different implemented operations

    input_vector = np.array([-0.27323645,0.12538062,0.95374082])
    indices = [0,2,1,4,2,2,3,3,3,2,2]
    output_vectors = np.array([[-0.6831809,-0.04200519,0.72904007],
                      [0.18289107,0.76098587,-0.62245591],
                      [-0.61517874,0.5147624,-0.59713884],
                      [-0.33867074,-0.80966534,-0.47931635],
                      [-0.52629529,-0.78190408,0.33412466]])

    output_words = output_vectors[indices,:]

    output_words_expected = np.array([[-0.6831809,-0.04200519,0.72904007],
                             [-0.61517874,0.5147624,-0.59713884],
                             [0.18289107,0.76098587,-0.62245591],
                             [-0.52629529,-0.78190408,0.33412466],
                             [-0.61517874,0.5147624 ,-0.59713884],
                             [-0.61517874,0.5147624 ,-0.59713884],
                             [-0.33867074,-0.80966534,-0.47931635],
                             [-0.33867074,-0.80966534,-0.47931635],
                             [-0.33867074,-0.80966534,-0.47931635],
                             [-0.61517874,0.5147624 ,-0.59713884],
                             [-0.61517874,0.5147624 ,-0.59713884]])

    assert np.allclose(output_words, output_words_expected, rtol=1e-05, atol=1e-06)

    K = 10

    directions = np.array([1] + [-1 for k in range(K)])
    delta, _ = sigmoid(np.dot(output_words, input_vector) * directions)
    delta_expected = np.array([0.70614176,0.5834337,0.63372281,0.40988622,
                               0.5834337 ,0.5834337,0.61446565,0.61446565,
                               0.61446565,0.5834337,0.5834337])

    assert np.allclose(delta, delta_expected, rtol=1e-05, atol=1e-06)

    delta_minus = (delta - 1) * directions;
    delta_minus_expected = np.array([-0.29385824,0.4165663,0.36627719,0.59011378,
                             0.4165663,0.4165663,0.38553435,0.38553435,
                             0.38553435,0.4165663,0.4165663])

    assert np.allclose(delta_minus, delta_minus_expected, rtol=1e-05, atol=1e-06)

    grad_pred = np.dot(delta_minus.reshape(1,K+1), output_words).flatten()
    grad_pred_expected = np.array([-1.71584818,-0.03463511,-2.0431726 ])

    assert np.allclose(grad_pred, grad_pred_expected, rtol=1e-05, atol=1e-06)

    N = np.shape(output_vectors)[1]
    grad_min = np.dot(delta_minus.reshape(K+1,1), input_vector.reshape(1,N))
    grad_min_expected = np.array([[ 0.08029278,-0.03684413,-0.28026459],
                         [-0.11382109,0.05222934,0.39729628],
                         [-0.10008028,0.04592406,0.34933351],
                         [-0.16124059,0.07398883,0.5628156 ],
                         [-0.11382109,0.05222934,0.39729628],
                         [-0.11382109,0.05222934,0.39729628],
                         [-0.10534204,0.04833854,0.36769985],
                         [-0.10534204,0.04833854,0.36769985],
                         [-0.10534204,0.04833854,0.36769985],
                         [-0.11382109,0.05222934,0.39729628],
                         [-0.11382109,0.05222934,0.39729628]])

    assert np.allclose(grad_min, grad_min_expected, rtol=1e-05, atol=1e-06)

    print("\033[92m" + "... end test" + "\033[0m")


def test_word2vec():
    print("\n" + "\033[92m" + "Test Word2Vec models with softmax and negative sampling gradients ..." + "\033[0m")
    dataset = type('dummy', (), {})() # It creates class dynamically and creates an instance of it

    # dummySampleTokenIdx() and getRandomContext(C) implemented here represent dummy functions.
    # A better implementation is available in applications/sentiment/stanford.py

    def dummySampleTokenIdx(): # It generates randomly an int between 0 and 4
        return random.randint(0, 4)

    # Example of output: ('b', ['c', 'a'])
    # Example of output: ('c', ['c', 'b', 'e', 'a', 'b', 'e'])
    def getRandomContext(C): # C is equal to the number of elements in the context (window)
        tokens = ["a", "b", "c", "d", "e"]
        return tokens[random.randint(0,4)], [tokens[random.randint(0,4)] for i in range(2*C)] # C is a window

    dataset.sampleTokenIdx = dummySampleTokenIdx
    dataset.getRandomContext = getRandomContext

    random.seed(31415)
    np.random.seed(9265)

    dummy_vectors = normalize_rows(np.random.randn(10,3))
    dummy_tokens = dict([("a",0), ("b",1), ("c",2),("d",3),("e",4)])

    print("\n==== Gradient check for skip-gram ====")
    gradient_check_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5), dummy_vectors)
    gradient_check_naive(lambda vec: word2vec_sgd_wrapper(skipgram, dummy_tokens, vec, dataset, 5, negative_sampling), dummy_vectors)

    print("\033[92m" + "... end test" + "\033[0m")


def test_word2vec_with_reg():
    print("\n" + "\033[92m" + "Test Word2Vec models with softmax and regularization ..." + "\033[0m")

    random.seed(314159)
    np.random.seed(265)

    dataset = StanfordSentiment()
    tokens = dataset.tokens()
    nWords = len(tokens)

    _, wordVectors0, _ = load_saved_params()
    wordVectors = wordVectors0[:nWords,:] + wordVectors0[nWords:,:]
    dimVectors = wordVectors.shape[1]

    dummy_weights = 0.1 * np.random.randn(dimVectors, 5)
    dummy_features = np.zeros((10, dimVectors))
    dummy_labels = np.zeros((10,), dtype=np.int32)

    for i in range(10):
        words, dummy_labels[i] = dataset.getRandomTrainSentence()
        dummy_features[i, :] = getSentenceFeature(tokens, wordVectors, words)

    print("==== Gradient check for softmax regression ====")
    gradient_check_naive(lambda weights: softmax_cost_grads_reg(dummy_features,
        dummy_labels, weights, 1.0, nopredictions = True), dummy_weights)

    print("\033[92m" + "... end test" + "\033[0m")


if __name__ == "__main__":
    test_normalize_rows()
    test_negative_sampling()
    test_word2vec()
    test_word2vec_with_reg()
