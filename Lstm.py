
import pandas as pd
import json
import csv
import re
import nltk
import numpy as np

input_data = pd.read_csv("dataset.csv")
dataset1 = {"reviewText": input_data["review_text"],  }
dataset = pd.DataFrame(data = dataset1)
dataset=dataset.fillna("Product was okay")
print(dataset.isnull().sum())
x = input_data.loc[:,"review_text"].values
y= input_data.loc[:,"good"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, test_size = 0.2, random_state = 0)
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

#word embedding and save it in txt format
model_ted = Word2Vec(sentences=corpus, size=100, window=5, min_count=5, workers=4, sg=0)
filename = 'reviews_embedding_word2vec.txt'
model_ted.wv.save_word2vec_format(filename, binary=False)
model_ted.wv.most_similar('good')

#to load the word embedding as a directory of words to vectors .we will extract word embeddings from a stored file 

import os

embeddings_index = {}
f = open(os.path.join('', 'reviews_embedding_word2vec.txt'),  encoding = "utf-8")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:])
    embeddings_index[word] = coefs
f.close()
print("works1")
#Calculating max length
total_reviews = x
max_length=0
#max_length = [len(s.split()) for s in total_reviews]
for s in total_reviews:
    a=str(s)
    p=len(a.split())
    if p > max_length:
        max_length=p

from tensorflow.python.keras.preprocessing.text import Tokenizer
from tensorflow.python.keras.preprocessing.sequence import pad_sequences

# vectorize the text samples into a 2D integer tensor
tokenizer_obj = Tokenizer()
tokenizer_obj.fit_on_texts(corpus)
sequences = tokenizer_obj.texts_to_sequences(corpus)

word_index = tokenizer_obj.word_index
print('Found %s unique tokens.' % len(word_index))

review_pad = pad_sequences(sequences, maxlen=max_length)
print("works22")

num_validation_samples = int(0.2* review_pad.shape[0])
X_train_pad=review_pad[:-num_validation_samples]
X_test_pad=review_pad[-num_validation_samples:]
print('Shape of X_train_pad tensor:', X_train_pad.shape)
print('Shape of X_test_pad tensor:', X_test_pad.shape)
#map embeddings from the loaded word2vec model for each word to tokenizor_obj.word_index vocab and 
#create a matrix of word vectors 
EMBEDDING_DIM =100
num_words = len(word_index) + 1
embedding_matrix = np.zeros((num_words, EMBEDDING_DIM))

for word, i in word_index.items():
    if i > num_words:
        continue
    embedding_vector = embeddings_index.get(word)
    if embedding_vector is not None:
        # words not found in embedding index will be all-zeros.
        embedding_matrix[i] = embedding_vector
        
from keras.models import Sequential
from keras.layers import Dense, Embedding, LSTM, GRU
from keras.layers.embeddings import Embedding
from keras.initializers import Constant

# define model embedding_matrix used as input to embedding layer,
#so trainable=False since ebedding is already learned
model = Sequential()
embedding_layer = Embedding(num_words,
                            EMBEDDING_DIM,
                            embeddings_initializer=Constant(embedding_matrix),
                            input_length=max_length,
                            trainable=False)
model.add(embedding_layer)
model.add(LSTM(units=32,  dropout=0.2, recurrent_dropout=0.2))
model.add(Dense(1, activation='relu'))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())

from keras.utils import to_categorical

model.fit(X_train_pad, y_train, batch_size=128, epochs=5, verbose=2)

loss, accuracy = model.evaluate(X_test_pad, y_test, batch_size=128)
print('Accuracy: %f' % (accuracy*100))

# serialize model to JSON
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
# serialize weights to HDF5
model.save_weights("model.h5")
print("Saved model to disk")

# load json and create model
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
from keras.models import model_from_json
loaded_model = model_from_json(loaded_model_json)
# load weights into new model
loaded_model.load_weights("model.h5")
print("Loaded model from disk")














