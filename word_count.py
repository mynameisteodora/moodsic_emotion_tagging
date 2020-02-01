#!/usr/bin/env python3
import emotions
import utils
import json
import pandas as pd
import csv
import tqdm
from tqdm.notebook import tqdm
from nltk import word_tokenize
from langdetect import detect, detect_langs, DetectorFactory
import pickle

def get_word_freq(stanzas):
    non_english_indices = []
    word_dictionary = {}

    for i in tqdm(range(len(stanzas))):

        # if detect(stanzas[i]) != 'en':
        #     non_english_indices.append(i)
        #     continue

        words = [w.lower() for w in word_tokenize(stanzas[i]) if w.isalpha()]

        for word in words:
            if word in word_dictionary.keys():
                word_dictionary[word] += 1
            else:
                word_dictionary[word] = 1

    return word_dictionary, non_english_indices

if __name__ == '__main__':
    stanzas = []
    with open('MOODSIC_LYRICS_DATASET_SORTED.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in tqdm(reader):
            for st in row['Lyrics'].split('\n'):
                if len(st) > 4:
                    stanzas.append(st)

    word_dictionary, non_english_indices = get_word_freq(stanzas)

    with open('./vocabulary.p', 'wb') as f:
        pickle.dump(word_dictionary, f)
    f.close()

    with open('./non_english_indices.p', 'wb') as f:
        pickle.dump(non_english_indices, f)
    f.close()
