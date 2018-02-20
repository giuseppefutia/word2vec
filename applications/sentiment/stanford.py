#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
import random


class StanfordSentiment:
    def __init__(self, path=None, tablesize = 1000000):
        if not path:
            path = "datasets/stanford-sentiment-tree-bank"

        self.path = path
        self.tablesize = tablesize


    def tokens(self):
        """
        Get tokens from sentences and extract occurrences

        Returns:
        tokens -- dict where the key is the token and the value is its id. Example: b'cleaving', 12512
        tokenfreq -- dict with the frequency of the token. Example: b'cleaving': 1
        wordcount -- total of words extracted by sentence (considering all occurrences)
        revtokens -- list of individual tokens
        """
        if hasattr(self, "_tokens") and self._tokens:
            return self._tokens

        tokens = dict()
        tokenfreq = dict()
        wordcount = 0
        revtokens = []
        idx = 0

        for sentence in self.sentences():
            for w in sentence:
                wordcount += 1
                if not w in tokens:
                    tokens[w] = idx
                    revtokens += [w]
                    tokenfreq[w] = 1
                    idx += 1
                else:
                    tokenfreq[w] += 1

        tokens["UNK"] = idx
        revtokens += ["UNK"]
        tokenfreq["UNK"] = 1
        wordcount += 1

        # Remember: len(tokens) is equals to len(revtokens)
        self._tokens = tokens
        self._tokenfreq = tokenfreq
        self._wordcount = wordcount
        self._revtokens = revtokens
        return self._tokens


    def sentences(self):
        """
        Get sentences and count them

        Returns:
        sentences -- list of sentences. Each element contains the tokens of the sentence. Example:
        [b'the', b'rock', b'is', b'destined', b'to', b'be', b'the', b'21st', b'century', b"'s", b'new', b'``', b'conan', b"''", b'and', b'that', b'he', b"'s", b'going', b'to', b'make', b'a', b'splash', b'even', b'greater', b'than', b'arnold', b'schwarzenegger', b',', b'jean-claud', b'van', b'damme', b'or', b'steven', b'segal', b'.']
        sentlengths -- np.array with the length of each sentence
        cumsentlen -- list with the cumulative sum of sentlengths
        """
        if hasattr(self, "_sentences") and self._sentences:
            return self._sentences

        sentences = []
        with open(self.path + "/dataset_sentences.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue
                splitted = line.strip().split()[1:]
                # Deal with some peculiar encoding issues with this file
                # sentences += [[w.lower().decode("utf-8").encode('latin1') for w in splitted]] -- Works only in Python 2
                sentences += [[w.lower().encode('latin1') for w in splitted]]

        self._sentences = sentences
        self._sentlengths = np.array([len(s) for s in sentences])
        self._cumsentlen = np.cumsum(self._sentlengths)

        return self._sentences


    def num_sentences(self):
        """
        Get the number of sentences

        Returns:
        num_sentences -- number of sentences
        """
        if hasattr(self, "_num_sentences") and self._num_sentences:
            return self._num_sentences
        else:
            self._num_sentences = len(self.sentences())
            return self._num_sentences


    def allSentences(self):
        """
        Purify sentences of all common words

        Returns:
        allSentences -- Returns all sentences "purified" of all commons words
        """
        if hasattr(self, "_allsentences") and self._allsentences:
            return self._allsentences

        # Example of a single sentence:
        # [b'the', b'rock', b'is', b'destined', b'to', b'be', b'the', b'21st', b'century', b"'s", b'new', b'``', b'conan', b"''", b'and', b'that', b'he', b"'s", b'going', b'to', b'make', b'a', b'splash', b'even', b'greater', b'than', b'arnold', b'schwarzenegger', b',', b'jean-claud', b'van', b'damme', b'or', b'steven', b'segal', b'.']
        sentences = self.sentences()
        # Example of rejectProb[0] 0.985020880779 in the case of the token b'the'
        rejectProb = self.rejectProb()
        # Example of tokens['b,the'] is 0 because it is the first word of all sentences
        tokens = self.tokens()

        allsentences = [[w for w in s
            if 0 >= rejectProb[tokens[w]] or random.random() >= rejectProb[tokens[w]]]
            for s in sentences * 30]

        allsentences = [s for s in allsentences if len(s) > 1]

        # Is sentence[0] =
        # [b'the', b'rock', b'is', b'destined', b'to', b'be', b'the', b'21st', b'century', b"'s", b'new', b'``', b'conan', b"''", b'and', b'that', b'he', b"'s", b'going', b'to', b'make', b'a', b'splash', b'even', b'greater', b'than', b'arnold', b'schwarzenegger', b',', b'jean-claud', b'van', b'damme', b'or', b'steven', b'segal', b'.']
        # All sentences[0] =
        # [b'be', b'century', b'conan', b'he', b'arnold', b'jean-claud', b'van', b'damme', b'steven', b'segal']

        self._allsentences = allsentences

        return self._allsentences


    def getRandomContext(self, C=5):
        """
        Extract the center word and the context to train a Word2Vec model

        Returns:
        centerword -- center word for skip-gram model
        context -- list of context words for skip-gram model
        """
        # Sentences purified by common words
        allsent = self.allSentences()
        # Randomly get one sentence
        sentID = random.randint(0, len(allsent) - 1)
        sent = allsent[sentID]
        # Randomly get one word of this sentence
        wordID = random.randint(0, len(sent) - 1)
        context = sent[max(0, wordID - C):wordID]

        # If the purified sentence is equals to
        # [b'sermonizing', b'lifeless', b'paean', b'teenage', b'dullards']
        # and the WordID is equals to 2
        # then context is equals to [b'sermonizing', b'lifeless']
        # This happens in case in which wordID is less than C (context window)

        if wordID+1 < len(sent):
            context += sent[wordID+1:min(len(sent), wordID + C + 1)]

        centerword = sent[wordID]

        context = [w for w in context if w != centerword]

         # Example to understand previous lines:
         # sent = [b'coarse', b'cliched', b'clunky', b'trifling', b'opposites', b'screenplay', b'demands', b'squanders', b'charms', b'sandra']
         # len(sent) = 10
         # if wordID == 6, then context is [b'clunky', b'trifling', b'opposites', b'screenplay']
         # Adding new words to context, I obtain: [b'clunky', b'trifling', b'opposites', b'screenplay', b'squanders', b'charms', b'sandra']
         # The center word is b'demands'
         # The filnal context is [b'clunky', b'trifling', b'opposites', b'screenplay', b'squanders', b'charms', b'sandra']

        if len(context) > 0:
            return centerword, context
        else:
            return self.getRandomContext(C)


    def sentiment_labels(self):
        """
        Get values of sentiment for each label

        Returns:
        sentiment_labels -- a list with sentiment values assigned to each sentence
        """
        if hasattr(self, "_sentiment_labels") and self._sentiment_labels:
            return self._sentiment_labels

        dictionary = dict()
        phrases = 0

        # dictionary.txt contains all phrases and their IDs, separated by a vertical line
        # Example: ! Brilliant !|40532

        with open(self.path + "/dictionary.txt", "r") as f:
            for line in f:
                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                # An item of dictionary has the form of ('a markedly inactive film , city is conversational bordering on confessional .', 103692)
                dictionary[splitted[0].lower()] = int(splitted[1])
                phrases += 1

        # Create a list of 0.0 elements of length equals to the number of sentences
        labels = [0.0] * phrases

        # sentiment_labels.txt contains all phrase ids and the corresponding sentiment labels, separated by a vertical line
        # Example: 12|0.33333
        with open(self.path + "/sentiment_labels.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                line = line.strip()
                if not line: continue
                splitted = line.split("|")
                # For each sentence in dictiorary you have an element in the labels list with the value of the sentiment converted in float
                labels[int(splitted[0])] = float(splitted[1])

        sentiment_labels = [0.0] * self.num_sentences()
        sentences = self.sentences()

        for i in range(self.num_sentences()):
            sentence = sentences[i]
            # XXX Some problem with utf-8 in Python 3
            full_sent = b" ".join(sentence).decode("latin-1").replace('-lrb-', '(').replace('-rrb-', ')').replace('ã©','é')

            # Added a try/catch because sometimes I can't get sentences as key for codification problems
            try:
                # sentiment labels is a list in which the index correspond to an a element of dictionary (a sentence) and the value of the element is the sentiment value for the specific sentence
                # Remember that with dictionary[full_sent] you get the ID of the sentence
                # With labels[dictionary[full_sent]] you get the sentiment value assigned to that sentence
                sentiment_labels[i] = labels[dictionary[full_sent]]
            except KeyError:
                 print("I got a KeyError - reason '%s'" % str(full_sent))

        self._sentiment_labels = sentiment_labels
        return self._sentiment_labels


    def dataset_split(self):
        """
        It splits the dataset in which the sentences are divided for training, testing, and developing

        Returns:
        split -- list of 3 elements. For each element you have a list of the indexes of sentences chosen for training, testing and developing
        """
        if hasattr(self, "_split") and self._split:
            return self._split

        split = [[] for i in range(3)]
        # dataset_split contains the sentence index (corresponding to the index in datasetSentences.txt file) followed by the set label separated by a comma
        # 1 = train, 2 = test, 3 = dev. Example: 1,1
        with open(self.path + "/dataset_split.txt", "r") as f:
            first = True
            for line in f:
                if first:
                    first = False
                    continue

                splitted = line.strip().split(",")
                split[int(splitted[1]) - 1] += [int(splitted[0]) - 1]

        self._split = split
        return self._split


    def getRandomTrainSentence(self):
        split = self.dataset_split()
        sentId = split[0][random.randint(0, len(split[0]) - 1)]
        return self.sentences()[sentId], self.categorify(self.sentiment_labels()[sentId])


    def categorify(self, label):
        if label <= 0.2:
            return 0
        elif label <= 0.4:
            return 1
        elif label <= 0.6:
            return 2
        elif label <= 0.8:
            return 3
        else:
            return 4


    def getDevSentences(self):
        return self.getSplitSentences(2)


    def getTestSentences(self):
        return self.getSplitSentences(1)


    def getTrainSentences(self):
        return self.getSplitSentences(0)


    def getSplitSentences(self, split=0):
        ds_split = self.dataset_split()
        return [(self.sentences()[i], self.categorify(self.sentiment_labels()[i])) for i in ds_split[split]]


    def sampleTable(self):
        """
        Create a list of self._tablesize elements with indexes of tokens repeated
        in such list according to implemented algorithm. Useful to randomly get
        elements for negative sampling

        Returns:
        sampleTable -- list of tokens indexes
        """
        if hasattr(self, '_sampleTable') and self._sampleTable is not None:
            return self._sampleTable

        nTokens = len(self.tokens())
        samplingFreq = np.zeros((nTokens,))
        self.allSentences()
        i = 0
        for w in range(nTokens):
            w = self._revtokens[i]
            if w in self._tokenfreq:
                freq = 1.0 * self._tokenfreq[w]
                # Reweigh
                freq = freq ** 0.75
            else:
                freq = 0.0
            samplingFreq[i] = freq
            i += 1

        samplingFreq /= np.sum(samplingFreq)
        samplingFreq = np.cumsum(samplingFreq) * self.tablesize

        self._sampleTable = [0] * self.tablesize

        # If self._revtokens[0] is equals to b'the',
        # then is frequency (self._tokenfreq[self._revtokens[0]]) is equals to 10128.
        # When I reweigh this value you obtain 1009.58472134.
        # I calculate the frequency percentage using samplingFreq /= np.sum(samplingFreq):
        # I obtain 0.0135592070799 (a sort of softmax without exponential).
        # Then I multiply this value for the tablesize (1000000) and I obtain 13559.2070799
        # and for each position I calculate the cumulative sum.

        j = 0
        for i in range(self.tablesize):
            while i > samplingFreq[j]:
                j += 1
            self._sampleTable[i] = j

        return self._sampleTable


    def rejectProb(self):
        """
        rewigh the probability of a token according to a specific threshold.
        If the frequency of the token overcome this threshold, then you assign a probability between 0 and 1.
        Otherwise the probability of the token is assigned to 0.

        Returns:
        rejectProb -- np.arry of probabilities values of each token rewighed according to a threshold

        """
        if hasattr(self, '_rejectProb') and self._rejectProb is not None:
            return self._rejectProb

        # If wordcount is 227246, the threshold is 2.27246
        threshold = 1e-5 * self._wordcount
        nTokens = len(self.tokens())
        # You define a np array of probabilities with length equals to the number of tokens
        rejectProb = np.zeros((nTokens,))

        for i in range(nTokens):
            # if revtokens[i] = b'the', the first item of tokenfreq is (b'the': 10128)
            w = self._revtokens[i]
            freq = 1.0 * self._tokenfreq[w]
            # Reweigh: # if the value of freq is less than the threshold (in other words if its frequency is less than 0.00001%) its probability is map to 0
            rejectProb[i] = max(0, 1 - np.sqrt(threshold / freq))

        self._rejectProb = rejectProb
        return self._rejectProb


    def sampleTokenIdx(self):
        return self.sampleTable()[random.randint(0, self.tablesize - 1)]
