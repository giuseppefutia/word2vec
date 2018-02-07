import random
from sentiment_utils import *
import sys
sys.path.insert(0, "./")
sys.path.insert(1, "./utils")
from word2vec import *
from stochastic_gradient_descent import *

random.seed(314)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

print("Number of words is equal to: " + str(n_words))

# Train 10-dimensional vectors
dim_vectors = 10

# Context size
C = 5

random.seed(31415)
np.random.seed(9265)

word_vectors = np.concatenate(((np.random.rand(n_words, dim_vectors) - .5) / \
	dim_vectors, np.zeros((n_words, dim_vectors))), axis=0)

print("")

word_vectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negative_sampling),
    word_vectors, 0.3, 40000, None, True, PRINT_EVERY=10)

print("Sanity check: cost at convergence should be around or below 10")
