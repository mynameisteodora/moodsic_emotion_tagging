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
import nltk

from keras.datasets import imdb
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, GlobalAveragePooling1D, Dense, LSTM, Flatten, Reshape, Dropout, BatchNormalization
from keras.wrappers.scikit_learn import KerasClassifier
from keras.utils import to_categorical

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
from emotions import VAD
from emotions import *

from nltk.sentiment.vader import SentimentIntensityAnalyzer
from keras.models import load_model
import tensorflow as tf

vader = SentimentIntensityAnalyzer()
new_lexicon = pickle.load(open('./new_lexicon.p', 'rb'))
important_tags = ['FW', 'NN', 'NNS', 'NNP', 'NNPS', 'RB', 'RBR', 'RBS', 'UH', 'VB', 'VBD', 'VBG', 'VBN', "VBP", 'VBZ']

Positive = {
    'Relaxed': VAD(0.68, -0.46, 0.06),
    'Joyful': VAD(0.66, 0.68, 0.35)
}

Neutral = {
    'Startled': VAD(-0.09, 0.65, -0.33),
    'Solemn': VAD(0.03, -0.32, -0.11)
}

Negative = {
    'Upset': VAD(-0.63, 0.30, -0.24),
    'Sad': VAD(-0.63, -0.27, -0.33)
}

Labels = {
    'Pos': Positive,
    'Neu': Neutral,
    'Neg': Negative
}

emos_to_idx = {
    'Relaxed': 0,
    'Solemn': 1,
    'Joyful': 2,
    'Startled': 3,
    'Upset': 4,
    'Sad': 5,
}

model = tf.keras.models.load_model('predictor.h5')

def extract_vad(text):
    vads = []
    for word, pos in text:
        if word in new_lexicon:
            if pos in important_tags:
                vads.append(2*new_lexicon[word])
            else:
                vads.append(new_lexicon[word])

    if len(vads) == 0:
#         print("Unidentifiable text = {}".format(text))
        vad = [0.0, 0.0, 0.0]
    else:
        vad = sum(vads) / (len(vads))
        vad = [vad.v, vad.a, vad.d]
    return vad

def process_datapoint(stanza):
    pol_scores = vader.polarity_scores(stanza)
    pols = [pol_scores['neg'], pol_scores['neu'], pol_scores['pos'], pol_scores['compound']]
    stanza = [w.lower() for w in word_tokenize(stanza) if w.isalpha()]
    stanza = nltk.pos_tag(stanza)
    vads = extract_vad(stanza)
    data_point = pols + vads

    return data_point

def res_to_label(result):
    cols = {0:"Pos", 1:"Neu", 2:"Neg"}
    return cols[result]

def predict_emotion(stanza):
    dp = process_datapoint(stanza)
    result = res_to_label(np.argmax(model.predict([dp])))
    vad = dp[4:]
    dp_emotion = VAD(vad[0], vad[1], vad[2]).closest(Labels[result])

    return dp_emotion

if __name__ == '__main__':
    colnames = ['UID', 'Artist', 'Song', 'Lyrics']
    lyrics = pd.read_csv('MOODSIC_LYRICS_DATASET_SORTED.csv', nrows=395888, names=colnames, header=1)

    tagged_emos = open('./MOODSIC_LYRICS_EMOTIONS_SENTIMENT_1.csv', 'w')
    tagged_emos.write("UID, Relaxed, Solemn, Joyful, Startled, Upset, Sad\n")
    emotions = np.zeros((len(lyrics), 6))
    for idx, song in tqdm(enumerate(lyrics['Lyrics'])):
        stanzas = song.split('\n\n')
        emos = Counter()

        for stanza in stanzas:
            emo = predict_emotion(stanza)
            emos[emo] += 1

        total = sum([v for v in emos.values()])

        for k,v in emos_to_idx.items():
            emotions[idx][v] = emos[k]/total


        tagged_emos.write("{0}, {1}, {2}, {3}, {4}, {5}, {6}\n".format(lyrics.loc[idx]['UID'], emotions[idx][0], emotions[idx][1],
                                                                          emotions[idx][2], emotions[idx][3],
                                                                          emotions[idx][4], emotions[idx][5]))
