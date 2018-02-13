#!/usr/bin/env python

import numpy as np

def getSentenceFeature(tokens, word_vectors, sentence):
    """
    Obtain the sentence feature for sentiment analysis by averaging its word vectors

    Arguments:
    tokens -- a dictionary that maps words to their indices in the word vector list
    word_vectors -- word vectors (each row) for all tokens
    sentence -- a list of words in the sentence of interest

    Returns:
    sentence_vector -- feature vector for the sentence
    """
    sentence_vector = np.zeros((word_vectors.shape[1],))

    for word in sentence:
        vector = word_vectors[tokens[word],:]
        sentence_vector += vector

    sentence_vector /= len(sentence)

    return sentence_vector


def accuracy(y, yhat):
    """ Precision for classifier """
    print(y.shape)
    print(yhat.shape)
    assert(y.shape == yhat.T.shape)
    return np.sum(y == yhat) * 100.0 / y.size
