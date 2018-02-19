#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.insert(0, "./")
sys.path.insert(1, "./utils")

from sent_utils import *
from stanford import *
from stochastic_gradient_descent import *
from word2vec import *

# Test the model on differnt values of regularization to reduce the overfitting
REGULARIZATION = [0.0, 0.000001, 0.00001, 0.0001, 0.001, 0.01, 0.1, 1]

# Load the dataset
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

# Load the word vectors trained using trainer.py
_, wordVectors0, _ = load_saved_params()
wordVectors = (wordVectors0[:n_words,:] + wordVectors0[n_words:,:])
dimVectors = wordVectors.shape[1]

# Load the train set
trainset = dataset.getTrainSentences()
nTrain = len(trainset)
trainFeatures = np.zeros((nTrain, dimVectors))
trainLabels = np.zeros((nTrain,), dtype=np.int32)
for i in range(nTrain):
    words, trainLabels[i] = trainset[i]
    trainFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

# Prepare dev set features
devset = dataset.getDevSentences()
nDev = len(devset)
devFeatures = np.zeros((nDev, dimVectors))
devLabels = np.zeros((nDev,), dtype=np.int32)
for i in range(nDev):
    words, devLabels[i] = devset[i]
    devFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

results = []

for regularization in REGULARIZATION:
    random.seed(3141)
    np.random.seed(59265)
    weights = np.random.randn(dimVectors, 5)
    print("Training for reg=%f" % regularization)

    # We will do batch optimization
    weights = sgd(lambda weights: softmax_wrapper(trainFeatures, trainLabels,
        weights, regularization), weights, 3.0, 10000, PRINT_EVERY=100)

    # Test on train set
    _, _, pred = softmax_cost_grads_reg(trainFeatures, trainLabels, weights)
    trainAccuracy = accuracy(trainLabels, pred)
    print("Train accuracy (%%): %f" % trainAccuracy)

    # Test on dev set
    _, _, pred = softmax_cost_grads_reg(devFeatures, devLabels, weights)
    devAccuracy = accuracy(devLabels, pred)
    print("Dev accuracy (%%): %f" % devAccuracy)

    # Save the results and weights
    results.append({
        "reg" : regularization,
        "weights" : weights,
        "train" : trainAccuracy,
        "dev" : devAccuracy})

# Print the accuracies

print("=== Recap ===")
print("Reg\t\tTrain\t\tDev")
for result in results:
    print("%E\t%f\t%f" % (
        result["reg"],
        result["train"],
        result["dev"]))

# Pick the best regularization parameters
BEST_REGULARIZATION = None
BEST_WEIGHTS = None

bestdev = 0
for result in results:
    if result["dev"] > bestdev:
        BEST_REGULARIZATION = result["reg"]
        BEST_WEIGHTS = result["weights"];
        bestdev = result["dev"];

# Test regularization results on the test set
testset = dataset.getTestSentences()
nTest = len(testset)
testFeatures = np.zeros((nTest, dimVectors))
testLabels = np.zeros((nTest,), dtype=np.int32)
for i in range(nTest):
    words, testLabels[i] = testset[i]
    testFeatures[i, :] = getSentenceFeature(tokens, wordVectors, words)

_, _, pred = softmax_cost_grads_reg(testFeatures, testLabels, BEST_WEIGHTS)
print("Best regularization value: %E" % BEST_REGULARIZATION)
print("Test accuracy (%%): %f" % accuracy(testLabels, pred))

# Make a plot of regularization vs accuracy
plt.plot(REGULARIZATION, [x["train"] for x in results])
plt.plot(REGULARIZATION, [x["dev"] for x in results])
plt.xscale('log')
plt.xlabel("regularization")
plt.ylabel("accuracy")
plt.legend(['train', 'dev'], loc='upper left')
plt.savefig("regularization-accuracy_img.png")
plt.show()
