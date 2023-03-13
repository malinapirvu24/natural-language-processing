import os
import sys
import math
import contractions
import string
import re
import numpy as np
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from os import listdir
from tqdm import tqdm
# nltk.download('wordnet')
# nltk.download('omw-1.4')
# nltk.download('words')


def english_words(phrase):
    words = set(nltk.corpus.words.words())
    result = ' '.join([w.strip() for w in nltk.wordpunct_tokenize(phrase) if w.lower() in words or not w.isalpha()])
    return result


def process_dataset(dataset_path, processed_path, max_review_length=150):
    for ds in ['/train/pos/', '/train/neg/', '/test/pos/', '/test/neg/']:
        dir_length = len(listdir(dataset_path + ds))
        prev = -1
        if os.path.exists(processed_path + ds):
            for idx, file in enumerate(listdir(processed_path + ds)):
                os.remove(processed_path + ds + file)
        for idx, file in enumerate(listdir(dataset_path + ds)):
            with open(f'{dataset_path}{ds}{file}', 'r', encoding='utf8') as f:
                review = f.read()
                expanded = []
                for word in review.split():
                    expanded.append(contractions.fix(word))

                expanded_review = ' '.join(expanded)
                sp = re.sub('[^A-Za-z]+', ' ', expanded_review)
                sp = sp.lower()
                sp = sp.split()
                tokenization = [word for word in sp if not word in stopwords.words('english')]
                lm= WordNetLemmatizer()
                final = [lm.lemmatize(word) for word in tokenization]
                final = english_words(' '.join(final))
                if len(final) >= max_review_length:
                    continue

                with open(f'{processed_path}{ds}{file}', 'w+', encoding='utf8') as p:
                    p.write(final)

            perc = math.floor((idx + 1) / dir_length * 100)

            if perc != prev and perc % 10 == 0:
                prev = perc
                print(f'{ds} {perc}%')


def plot_histogram(path):
    frequency_array = np.zeros(801)
    for ds in ['/train/pos/', '/train/neg/', '/test/pos/', '/test/neg/']:
        dir_length = len(listdir(path + ds))
        lista = listdir(path + ds)
        for idx, file in tqdm(enumerate(lista)):
            with open(f'{path}{ds}{file}', 'r', encoding='utf8') as f:
                review = f.read()
                review = review.split(' ')
                frequency_array[len(review)]+=1


    plt.plot(frequency_array)
    plt.show()




