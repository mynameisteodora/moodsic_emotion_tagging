import numpy as np
from nltk.tokenize import TweetTokenizer
from nltk.stem.porter import PorterStemmer
from nltk.corpus import stopwords
from nltk import word_tokenize
from sklearn.preprocessing import MinMaxScaler

def preprocess_text(text):
    stemmer = PorterStemmer()
    stop_words = set(stopwords.words("english"))
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha()]
#     words_st = [w for w in words if not w in stop_words]
    stems = [stemmer.stem(w) for w in words]
    
    return stems

def tokenise_text(text):
    words = word_tokenize(text)
    words = [w.lower() for w in words if w.isalpha()]
    return words

def preprocess_dataset(dataset):
    dataset['preprocessed_text'] = [preprocess_text(text) for text in dataset['text']]
    dataset['tokenised_text'] = [tokenise_text(text) for text in dataset['text']]
    og_len = len(dataset)
    dataset = dataset[dataset['preprocessed_text'].map(lambda d: len(d)) > 0].reset_index()
    dataset = dataset.drop('index', axis=1)
    new_len = len(dataset)
    print("{} empty datapoints removed".format(og_len - new_len))
    return dataset

def normalise_VAD(emobank, feature_range=(-1,1)):
    scaler = MinMaxScaler(feature_range=(-1, 1))
    emobank['V_scaled'] = scaler.fit_transform(np.array(emobank['V']).reshape(-1, 1))
    emobank['A_scaled'] = scaler.fit_transform(np.array(emobank['A']).reshape(-1, 1))
    emobank['D_scaled'] = scaler.fit_transform(np.array(emobank['D']).reshape(-1, 1))
    
    return emobank