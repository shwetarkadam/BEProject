# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""


# Part 1 - Data Preprocessing


# Importing the libraries

import pandas as pd
import json
import csv
import re
import nltk


input_data = pd.read_csv("/home/shweta/Desktop/project/Electronics_5.csv")
#dataset = {"reviewText": input_data["reviewText"]  }
#dataset1 = pd.DataFrame(input_data,columns=['reviewText'], ['overall'] )
dataset1 = {"reviewText": input_data["reviewText"], "overall": input_data["overall"]  }
dataset = pd.DataFrame(data = dataset1)
dataset=dataset.fillna("Product was okay and it works")
print(dataset.isnull().sum())

"""data['Sentiment']"""
dataset = dataset[dataset["overall"] != '3']
dataset["Sentiment"] = dataset["overall"].apply(lambda rating : 1 if rating > '3' else 0)

y=dataset["Sentiment"].values


print("works")
# Cleaning the texts
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus = []

#for i in range(1, 1689188):
#for index, row in dataset.iterrows():
for i in range(1,27000):
    #review = re.su1b("[^a-zA-Z]", " ", row['reviewText'])
    review = re.sub("[^a-zA-Z]", " ", dataset['reviewText'][i])
    review = review.lower()
    review = review.split()
    ps = PorterStemmer()
    review = [ps.stem(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

# Creating the Bag of Words model
"""from sklearn.feature_extraction.text import TfidfVectorizer
vector = TfidfVectorizer(max_features = 1500)
X = vector.fit_transform(corpus).toarray()
Y = dataset.iloc[:, 1].values
"""
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import tenserflow as tf

tk = Tokenizer(lower = True)
tk.fit_on_texts(corpus)
X_seq = tk.texts_to_sequences(corpus)
X_pad = pad_sequences(X_seq, maxlen=100, padding='post')
print("hello")
print("Done")
print("Done")

#split into train and test sets
from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X_pad, y, test_size = 0.25, random_state = 1)