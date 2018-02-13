import random
import time
from stanford import *
import sys
sys.path.insert(0, "./")
from word2vec import *
sys.path.insert(1, "./utils")
from stochastic_gradient_descent import *

tic = time.clock()

random.seed(314, version=1)
dataset = StanfordSentiment()
tokens = dataset.tokens()
n_words = len(tokens)

print("Number of words is equal to: " + str(n_words))

# Train 10-dimensional vectors
dim_vectors = 10

# Context size
C = 5

random.seed(31415, version=1)
np.random.seed(9265)

word_vectors = np.concatenate(((np.random.rand(n_words, dim_vectors) - .5) / \
	dim_vectors, np.zeros((n_words, dim_vectors))), axis=0)

word_vectors0 = sgd(
    lambda vec: word2vec_sgd_wrapper(skipgram, tokens, vec, dataset, C, negative_sampling),
    word_vectors, 0.3, 40000, None, True, PRINT_EVERY=10)

print("Sanity check: cost at convergence should be around or below 10")

toc = time.clock()

print("Training time: " + str(toc-tic))

# Visualize the word vectors you trained
_, wordVectors0, _ = load_saved_params()

wordVectors = (wordVectors0[:n_words,:] + wordVectors0[n_words:,:])

# Binary strings defined in Python 3.
visualizeWords = [b"the", b"a", b"an", b",", b".", b"?", b"!", b"``", b"''", b"--",
	b"good", b"great", b"cool", b"brilliant", b"wonderful", b"well", b"amazing",
	b"worth", b"sweet", b"enjoyable", b"boring", b"bad", b"waste", b"dumb",
	b"annoying", b'immaculately']

visualizeIdx = [tokens[word] for word in visualizeWords]
visualizeVecs = wordVectors[visualizeIdx, :]

temp = (visualizeVecs - np.mean(visualizeVecs, axis=0))
covariance = 1.0 / len(visualizeIdx) * temp.T.dot(temp)
U,S,V = np.linalg.svd(covariance)
coord = temp.dot(U[:,0:2])

for i in range(len(visualizeWords)):
    plt.text(coord[i,0], coord[i,1], visualizeWords[i],
    	bbox=dict(facecolor='green', alpha=0.1))

plt.xlim((np.min(coord[:,0]), np.max(coord[:,0])))
plt.ylim((np.min(coord[:,1]), np.max(coord[:,1])))

plt.savefig('word_vectors.png')
plt.show()
