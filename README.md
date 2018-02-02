# Word2Vec
Implementation of the Word2Vec models.

## Shallow Neural Network
The Word2Vec implementation in this repository is based on the general purpose neural network available in `dnn.py` file. In order to test the network (forward and back propagation) you can launch the following command:

```
python dnn.py
```

## Word2Vec Implementation Details
Only the skip-gram model is currently available in this repository. In order to test the backward propagation implemented within such model with the gradient check, you can launch the following command:

```
python word2vec.py
```

## Code References
The code developed in this repository is based on the source code released in the following online courses:
* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Coursera
* [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/) - Stanford
