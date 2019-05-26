import re
import numpy as np
import pandas as pd
import tensorflow as tf
from collections import defaultdict
from nltk.tokenize import word_tokenize, sent_tokenize
from tensorflow.python.platform import gfile
import gensim.models.keyedvectors as word2vec
try:
  # pylint: disable=g-import-not-at-top
  import cPickle as pickle
except ImportError:
  # pylint: disable=g-import-not-at-top
  import pickle



def clean_str(string):
    """
    Tokenization/string cleaning for all datasets except for SST.
    Original taken from https://github.com/yoonkim/CNN_sentence/blob/master/process_data.py
    """
    string = re.sub(r"[^A-Za-z0-9(),!?\'\`]", " ", string)
        
    string = re.sub(r"\'s", " \'s", string)
    string = re.sub(r"\'ve", " \'ve", string)
    string = re.sub(r"n\'t", " n\'t", string)
    string = re.sub(r"\'re", " \'re", string)
    string = re.sub(r"\'d", " \'d", string)
    string = re.sub(r"\'ll", " \'ll", string)
    string = re.sub(r",", " , ", string)
    string = re.sub(r"!", " ! ", string)
    string = re.sub(r"\(", " \( ", string)
    string = re.sub(r"\)", " \) ", string)
    string = re.sub(r"\?", " \? ", string)
    string = re.sub(r"\s{2,}", " ", string)
    return string.strip().lower()

def load_data_and_labels(path):
    """
    Loads MR polarity data from files, splits the data into words and generates labels.
    Returns split sentences and labels.
    """
    # Load data from files
    dataset = np.array(pd.read_excel(path))
    positive = []
    negative = []
    for data in dataset:
        if data[1] == 1:
            positive.append(data[0])
        else:
            negative.append(data[0])
    positive = [s.strip() for s in positive]
    negative = [s.strip() for s in negative]

    x_text = positive + negative
    x_text = [clean_str(sent) for sent in x_text]

    # # Generate labels
    positive_labels = [[0, 1] for _ in positive]
    negative_labels = [[1, 0] for _ in negative]
    y = np.concatenate([positive_labels, negative_labels], 0)
    return [x_text, y]

def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int((len(data)-1)/batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]

def load_glove_vec(path):

    glove_wordmap = {}
    with open(path, "r", encoding ="utf8") as glove:
        for line in glove:
            name, vector = tuple(line.split(" ", 1))
            glove_wordmap[name] = np.fromstring(vector, sep=" ")
    return glove_wordmap


def restore(filename):
    vocab = {}
    with open(filename, "r", encoding='utf8') as word:
        for line in word:
            name, vector = tuple(line)
            vocab[name] = np.fromstring(vector, sep=" ")
    return vocab

def testpreprocess(x_text, vocab):
    sentences = sent_tokenize(x_text)
    processed = []

    for sentence in sentences:
        tokens = word_tokenize(sentence.lower())
        rows = []
        for i in range (137):
            if i < len(tokens):
                word=tokens[i]
                if word in vocab:
                    rows.append(vocab[word])
                else:
                    vector = np.random.uniform(-0.25, 0.25, 50)
                    vocab[word] = vector
                    rows.append(vector)
            else:
                pad = [0 for i in range (50)]
                rows.append(pad)
        processed.append(rows)
    return processed

def glovevec(x_text, max_document_length):
    glove_wordmap = load_glove_vec("glove.6B.50d.txt")

    processed = []
    words = {}
    for sentence in x_text:
        tokens = word_tokenize(sentence.lower())
        rows = []
        for i in range(max_document_length):
            if i < len(tokens):
                word = tokens[i]
                if word in glove_wordmap:
                    rows.append(glove_wordmap[word])
                    words[word] = glove_wordmap[word]
                else:
                    vector = np.random.uniform(-0.25, 0.25, 50)
                    glove_wordmap[word] = vector
                    words[word] = vector
                    rows.append(vector)
            else:
                pad = [0 for i in range (50)]
                rows.append(pad)
        processed.append(rows)
    return processed, words
    
def wordtovec(x_text, max_document_length):
    model = word2vec.KeyedVectors.load_word2vec_format("./GoogleNews-vectors-negative300.bin", binary=True)
    print("Google Word2Vec loaded")
    output = []
    vocab = defaultdict()
    for text in x_text:
        process = word_tokenize(text)
        vector = []
        for i in range (max_document_length):
            if i < len(process):
                vocab[process[i]] += 1
                vector.append(model[process[i]])
            else:
                pad = [0 for i in range (300)]
                vector.append(pad)
        output.append(vector)
    output = np.array(output)
    return output, vocab
