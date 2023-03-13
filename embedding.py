import spacy
import os
import math
import json
import numpy as np
import gensim
from os import listdir
from gensim.models import KeyedVectors

nlp = spacy.load('en_core_web_lg')


# model = KeyedVectors.load_word2vec_format('data/GoogleGoogleNews-vectors-negative300.bin', binary=True)

def unique_words(path):
    unique_words = set()
    for ds in ['/train/pos/', '/train/neg/', '/test/pos/', '/test/neg/']:
        prev = -1
        dir_length = len(listdir(path + ds))

        for idx, file in enumerate(listdir(path + ds)):
            with open(f'{path}{ds}{file}', 'r', encoding='utf8') as f:
                review = f.read()
                for s in review.split():
                    unique_words.add(s)

            perc = math.floor((idx + 1) / dir_length * 100)
            if perc != prev:
                prev = perc
                print(f'{ds} {perc}%')

    return unique_words


def create_dictionary(processed_path):
    default_value = np.zeros(300)
    dictionary_keys = unique_words(processed_path)
    dictionary = {}
    for key in dictionary_keys:
        print(key)
        try:
            dictionary[key] = nlp(key).vector.tolist()
        except KeyError:
            dictionary[key] = default_value.tolist()

    with open('dictionary.json', 'w') as json_file:
        json.dump(dictionary, json_file)
