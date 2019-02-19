#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 19 15:27:36 2019

@author: shweta
"""


# Importing the libraries

import pandas as pd
import json
import csv
import re
import nltk


input_data = pd.read_csv("/home/shweta/Desktop/project/dataset.csv")
dataset1 = {"reviewText": input_data["review_text"],  }
dataset = pd.DataFrame(data = dataset1)
dataset=dataset.fillna("Product was okay")
print(dataset.isnull().sum())

# Cleaning the texts
import re
import nltk
from nltk.stem.porter import PorterStemmer
corpus = []

#for i in range(1, 1689188):
#for index, row in dataset.iterrows():
for i in range(1,30000):
    #review = re.su1b("[^a-zA-Z]", " ", row['reviewText'])
    review = re.sub("[^a-zA-Z]", " ", dataset['reviewText'][i])
    review = review.lower()
    review = review.split()    
    corpus.append(review)
print("works")

#word2vec converting words to vectors?word embedding 
from gensim.models import Word2Vec

model_ted = Word2Vec(sentences=corpus, size=100, window=5, min_count=5, workers=4, sg=0)
model_ted.wv.most_similar('good')

from gensim.test.utils import common_texts, get_tmpfile
path = get_tmpfile("/home/shweta/Desktop/project/word2vec.model")
model_ted.save("/home/shweta/Desktop/project/word2vec.model")