#!/usr/bin/env python3

from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import *
from nltk.corpus import stopwords
from nltk import word_tokenize

from collections import Counter, defaultdict
from sklearn.cluster import KMeans
from keras.utils import np_utils
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import MinMaxScaler
from sklearn.utils import class_weight
from sklearn import metrics

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
from spellchecker import SpellChecker

import pandas as pd
import numpy as np

scaler = MinMaxScaler(feature_range=(-1, 1))
def scale(arr):
    return scaler.fit_transform(np.array(arr).reshape(-1, 1))

if __name__ == '__main__':

    print("Reading lexicon...")
    lexicon = pd.read_csv('./NRC-VAD-Lexicon.txt', sep='\t')
    lexicon.columns = ['Word', 'V', 'A', 'D']
    # transform lexicon into a dictionary
    stemmer = PorterStemmer()
    words = list(lexicon['Word'])
    words = [stemmer.stem(str(word)) for word in words]
    wordV = scale(list(lexicon['V']))
    wordA = scale(list(lexicon['A']))
    wordD = scale(list(lexicon['D']))
    lexicon = dict([word, VAD(wordV[idx][0], wordA[idx][0], wordD[idx][0])] for idx, word in enumerate(words))

    # # turn the lexicon from VAD to emotions
    # print("Turning them into emotions...")
    # for word, vad in lexicon.items():
    #     emo = vad.closest(emotions.Final)
    #     lexicon[word] = emo

    # Add synonyms
    print("Adding synonyms...")
    new_lexicon = {}

    for word, emo in lexicon.items():
        diff_emos = 0
        # check first and second synonym, if they exist
        syn = wordnet.synsets(word)

    #     print("Word = {}".format(word))

        if len(syn) > 0:
            # take the first word
            first_syn = syn[0].lemmas()[0].name()
    #         print("Syn = {}".format(first_syn))

            if first_syn in lexicon.keys():
                if lexicon[first_syn] != emo:
                    diff_emos += 1

            new_lexicon[first_syn] = emo
            new_lexicon[word] = emo
        else:
            new_lexicon[word] = emo

    print("{} words added".format(len(new_lexicon) - len(lexicon)))
    print("{} emotions differred".format(diff_emos))

    # load the vocabulary and remove the stopwords
    print("Loading vocabulary...")
    vocabulary = pickle.load(open('./vocabulary.p', 'rb'))
    print("Length of vocab = {}".format(len(vocabulary)))

    vocabulary = {k: v for k, v in sorted(vocabulary.items(), key=lambda item: item[1], reverse=True)}

    print("Cleaning vocabulary...")
    stop_words = set(stopwords.words('english'))
    clean_vocabulary = {}
    for word in vocabulary.keys():
        if word not in stop_words:
            clean_vocabulary[word] = vocabulary[word]

    print("Removed {} items".format(len(vocabulary) - len(clean_vocabulary)))

    # see which top words are not in the lexicon and need further tagging
    need_tags = {}
    for word in clean_vocabulary.keys():
    #     print(word)
        if word not in new_lexicon.keys():
            need_tags[word] = clean_vocabulary[word]

    print("{} words still need tagging".format(len(need_tags)))

    # For each word that needs tagging, check for its correct spelling.
    # If its correct spelling is already in the lexicon, tag this word with that
    # emotion and add the incorrect spelling to the lexicon.
    spell = SpellChecker()

    print("Checking for spelling corrections")
    for word in tqdm(need_tags.keys()):
        new_words = 0
        remaining_words = []
        spelling = spell.unknown([word])

        if len(spelling) > 0:
            # this is misspelled
            correct_spelling = spell.correction(word)

            # if this spelling is in the original lexicon, get the emotion
            if correct_spelling in new_lexicon.keys():
                emo = new_lexicon[correct_spelling]

                # add the incorrect spelling to the lexicon with the same emotion
                new_lexicon[word] = emo
                new_words += 1
            else:
                remaining_words.append(word)

    print("New words added = {}".format(new_words))

    with open('./new_lexicon.p', 'wb') as f:
        pickle.dump(new_lexicon, f)
    f.close()

    with open('./remaining_words.p', 'wb') as f:
        pickle.dump(remaining_words, f)
    f.close()
