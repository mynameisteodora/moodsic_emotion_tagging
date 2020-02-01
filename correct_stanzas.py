#!/usr/bin/env python3
import emotions
import utils
import json
import pandas as pd
import csv
import tqdm
from tqdm import tqdm
from nltk import word_tokenize
from langdetect import detect, detect_langs, DetectorFactory
import pickle
from spellchecker import SpellChecker

def correct_stanzas(stanzas):
    x = 0
    non_english_indices = []

    for i in tqdm(range(len(stanzas))):

#         if detect(stanzas[i]) != 'en':
#             non_english_indices.append(i)
#             continue

        words = [w.lower() for w in word_tokenize(stanzas[i]) if w.isalpha()]
        misspelled = spell.unknown(words)
#         print(misspelled)

        if len(misspelled) > 0:
            for j in range(len(words)):
                if words[j] in misspelled:
                    words[j] = spell.correction(words[j])

            stanzas[i] = " ".join(words)

        if i % 10000 == 0:
            print("{} 10K...".format(x))
            x+= 1

    return stanzas

if __name__ == '__main__':
    print("Loading profanities...")
    with open('./profanities.txt') as file:
        profanities = []
        for line in file:
            words = line.split(', ')
            profanities.extend(words)

    spell = SpellChecker()
    spell.word_frequency.load_words(profanities)

    print("Loading stanzas...")
    one = pickle.load(open('./one.p', 'rb'))

    print("Spellchecking")
    correct_stanzas_one = correct_stanzas(one)

    with open('./correct_stanzas_one.p', 'wb') as f:
        pickle.dump(correct_stanzas_one, f)
