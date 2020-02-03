#!/usr/bin/env python3
import keras
import pandas as pd
import numpy as np
import re
import string
import csv

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import word_tokenize

from keras.datasets import imdb
from tensorflow.keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.layers import Embedding, GlobalAveragePooling1D, Dense, LSTM, Flatten, Reshape, Dropout
from keras.wrappers.scikit_learn import KerasClassifier

from matplotlib import pyplot as plt
from mpl_toolkits import mplot3d

import os

os.environ['KMP_DUPLICATE_LIB_OK']='True'

from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn import metrics
from sklearn.metrics import confusion_matrix

import seaborn as sns
from tqdm import tqdm
from sklearn.metrics import f1_score
import json

import pickle

import utils
import emotions
from emotions import VAD

import nltk
from nltk.corpus import wordnet

important_tags = ['FW', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', "VBP", 'VBZ']
new_lexicon = pickle.load(open('./new_lexicon.p', 'rb'))
Final = {
    'Relaxed': VAD(0.68, -0.46, 0.06),
    'Solemn': VAD(0.03, -0.32, -0.11),
    'Joyful': VAD(0.76, 0.48, 0.35),
    'Startled': VAD(-0.09, 0.65, -0.33),
    'Upset': VAD(-0.63, 0.30, -0.24),
    'Sad': VAD(-0.63, -0.27, -0.33),
}

emos_to_idx = {
    'Relaxed': 0,
    'Solemn': 1,
    'Joyful': 2,
    'Startled': 3,
    'Upset': 4,
    'Sad': 5,
}

def predict(text):
    vads = []
    for word, pos in text:
        if word in new_lexicon:
            w = weights[new_lexicon[word].closest(Final)]
            if pos in important_tags:
                vads.append(w*2*new_lexicon[word])
            else:
                vads.append(w*new_lexicon[word])
    #     vads = [weights[new_lexicon[word].closest(Final)] * new_lexicon[word] for (word, pos) in text
    #             if word in new_lexicon and pos in important_tags]
    if len(vads) == 0:
    #         print("Unidentifiable text = {}".format(text))
        return 'None'
    else:
        vad = sum(vads) / (len(vads))
        return vad.closest(Final)

if __name__ == '__main__':

    c = Counter()
    for l, vad in new_lexicon.items():
        c[vad.closest(Final)] += 1
    print(c)
    total = sum([v for v in c.values()])

    weights = {}
    for emo, count in c.items():
        weights[emo] = -1 * np.log(count / total)

    print("Weights new lexicon = {}".format(weights))

    lyrics = pd.read_csv('MOODSIC_LYRICS_DATASET_SORTED.csv', nrows=395888)

    tagged_emos = open('./MOODSIC_LYRICS_EMOTIONS.csv', 'w')
    tagged_emos.write("UID, Relaxed, Solemn, Joyful, Startled, Upset, Sad\n")
    emotions = np.zeros((len(lyrics), 6))
    for idx, song in tqdm(enumerate(lyrics['Lyrics'])):
        stanzas = song.split('\n')
        emos = Counter()
        for st in stanzas:
            text = [w.lower() for w in word_tokenize(st) if w.isalpha()]
            text = nltk.pos_tag(text)
            emo = predict(text)
            emos[emo] += 1

        # sum all emotions in counter
        total = sum([v for v in c.values()])

        for k,v in emos_to_idx.items():
            emotions[idx][v] = emos[k]/total


        tagged_emos.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(lyrics.loc[idx]['UID'], emotions[idx][0], emotions[idx][1],
                                                                      emotions[idx][2], emotions[idx][3],

    tagged_emos.close()
