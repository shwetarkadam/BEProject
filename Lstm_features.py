# -*- coding: utf-8 -*-
"""
Created on Fri Feb 22 08:56:42 2019

@author: HP
"""

import pandas as pd
import json
import csv
import re
import nltk
import numpy as np

input_data = pd.read_csv("dataset.csv")
#x = {"Count_Reviews": input_data["Count_Reviews"], "overall": input_data["overall"], "product_Popularity": input_data["product_Popularity"]}

x_21 = input_data.loc[:, "Count_Reviews"]
x_22 = input_data.loc[:, "overall"]
x_23 = input_data.loc[:, "product_Popularity"]

x_2 = []

for i in range(0, len(x_21)):
    x_2.append([x_21[i], x_22[i], x_23[i]]) 
    

y_2 = input_data.loc[:,"good"].values

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_2, y_2, test_size = 0.2, random_state = 0)
 

#X_train = np.reshape(X_train, (24000, 3))



# define model embedding_matrix used as input to embedding layer,
#so trainable=False since ebedding is already learned
# Part 2 - Building the RNN
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Dropout

#Intitialising the rnn
model=Sequential()

# Adding the first LSTM layer and some Dropout regularisation
model.add(LSTM(units = 50, return_sequences = True, input_shape = (24000,3)))

# try using different optimizers and different optimizer configs
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

print('Summary of the built model...')
print(model.summary())

from keras.utils import to_categorical

model.fit(X_train, y_train, batch_size=128, epochs=5, verbose=2)

loss, accuracy = model.evaluate(X_test, y_test, batch_size=128)
print('Accuracy: %f' % (accuracy*100))