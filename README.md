# Word2Vec
Implementation of the Word2Vec Skip-Gram model.

## System Requirements

* python 3.6
* conda 4.4.8

## Shallow Neural Network
The Word2Vec implementation in this repository is based on the general purpose neural network available in `dnn.py` file. In order to test the network (forward and back propagation) you can launch the following command:

```
python tests/dnn_test.py 
```

As mentioned before, the implemented neural network can be used for different purposes. A simple example on image classification can be test with the following command:

```
python applications/image_classifier.py
```

Such application exploits 2 different datasets (one for the training step and the other for the test step):

```
datasets/train_catvnoncat.h5
datasets/test_catvnoncat.h5
```

After 2500 iterations at the training stage, you should obtain the following accuracy results:

* **Accuracy: 1.0 (on the training dataset)**
* **Accuracy: 0.72 (on the test dataset)**


## Word2Vec Implementation 
Only the Skip-Gram model is currently available in this repository. In order to test with gradient check the backward propagation implemented within such model, you can launch the following command:

```
python tests/word2vec_test.py 
```

The model has been trained using the [Sentiment Treebank](https://nlp.stanford.edu/sentiment/treebank.html) dataset built by the Stanford NLP Group. Datasets has been published in this repository:

```
datasets/stanford-sentiment-tree-bank/
```

To train the model you have to launch the following command:

```
python applications/sentiment/trainer.py
```

Cost at convergence should be around or below 10.

Below a vector representation of some words after the training stage:

![Word vectors](https://github.com/giuseppefutia/word2vec/blob/master/word_vectors.png)

To test the accuracy of the trained model using regularization, you can launch the following command:

```
python applications/sentiment/test.py 
```

This test is performed is used exploiting different version of regularization. With the developed model you obtain the following results in terms of accuracy.

|Regularization|Train Acc|Dev Acc|
|--------------|---------|-------|
|0.000000E+00  |28.441011|29.700272
|1.000000E-06  |28.441011|29.881926
|1.000000E-05  |28.511236|29.609446
|1.000000E-04  |28.359082|28.065395
|1.000000E-03  |27.141854|25.340599
|1.000000E-02  |27.153558|25.522252
|1.000000E-01  |27.153558|25.522252
|1.000000E+00  |13.096910|13.079019


![Accuracy based on different values of regularization](https://github.com/giuseppefutia/word2vec/blob/master/regularization-accuracy_img.png)

## Code References
The code developed in this repository is based on the source code released in the following online courses:
* [Deep Learning Specialization](https://www.coursera.org/specializations/deep-learning) - Coursera
* [Deep Learning for Natural Language Processing](http://cs224d.stanford.edu/) - Stanford (converted to enable the use in Python 3)
