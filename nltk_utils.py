
import nltk
import ssl

import numpy as np
from nltk.stem.porter import PorterStemmer

stemmer = PorterStemmer()


def tokenize(sentence):
    return nltk.word_tokenize(sentence)


def stem(word):
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, all_words):
    bag = np.zeros(len(all_words),dtype=np.float32)
    tokenized_sentence = [stem(word) for word in tokenized_sentence]

    for idx,w in enumerate(all_words):
        if w in tokenized_sentence:
            bag[idx]=1

    return bag
