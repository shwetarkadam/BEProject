# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.
"""

import pandas as pd
import numpy as np

# create a vocabulary list containing distinct word
def createVocabDict(review_text):
    vocabSet = set()
    for review in review_text:
        for word in review.split():
                vocabSet.add(word)
    vocabDict = {}
    idx = 0
    for word in vocabSet:
        vocabDict[word] = idx
        idx += 1
    return vocabDict

# create Word vector for a given review
def createWordVector(vocabDict, review):
    wordVec = [0]*len(vocabDict)
    try:
        for word in review.split():
            wordVec[vocabDict[word]] = 1
    except KeyError as e:
        print(word, e)
    except ValueError as e:
        print(word, e)
    return np.array(wordVec)

# Splliting data into training and test data
# mediator decides how much should be used for training data => (0, 1)
def train_test_split(data, classification, mediator):
    len_data = len(data)
    len_train_data = int(mediator * len_data)
    
    train_data = data[:len_train_data]
    train_class = classification[:len_train_data]
    
    test_data = data[len_train_data:]
    test_class = classification[len_train_data:]
    
    return train_data, train_class, test_data, test_class


def train_naive_bayes(train_data_matrix, train_class, vocabDict):
    n = len(vocabDict)
    len_train_class = len(train_class)
    none, good, bad = 1, 1, 1
    
    for classfication in train_class:
        if (classfication == [0, 0]):
            none += 1
        elif (classfication == [1, 0]):
            good += 1
        elif (classfication == [0, 1]):
            bad += 1
    
    print(none, good, bad)
    
    pNoneTotal = float(none) / len_train_class
    pGoodTotal = float(good) / len_train_class
    pBadTotal = float(bad) / len_train_class
    
    noneReviewWords = np.array([1]*n)
    goodReviewWords = np.array([1]*n)
    badReviewWords = np.array([1]*n)
    
    for i in range(0, len(train_data_matrix)):
        print(train_class[i])
        if (train_class[i] == [0, 0]):
            noneReviewWords += train_data_matrix[i]
        elif (train_class[i] == [1, 0]):
            goodReviewWords += train_data_matrix[i]
        elif (train_class[i] == [0, 1]):
            badReviewWords += train_data_matrix[i]
            
    print(noneReviewWords, goodReviewWords, badReviewWords)
    
    pNoneVector = np.log(noneReviewWords / (np.sum(noneReviewWords) + 2))
    pGoodVector = np.log(goodReviewWords / (np.sum(goodReviewWords) + 2))
    pBadVector = np.log(badReviewWords / (np.sum(badReviewWords) + 2))
    
    return dict(pNoneVector=pNoneVector,
            pNoneTotal=pNoneTotal,
            pGoodVector=pGoodVector,
            pGoodTotal=pGoodTotal,
            pBadVector=pBadVector,
            pBadTotal=pBadTotal
            )
    #return [noneReviewWords, goodReviewWords, badReviewWords]


def classifier(test_review, test_class, nb):
    print(test_class)
    test_review_vector = np.array(test_review[i])
    pNone = np.sum(test_review_vector * nb['pNoneVector']) + np.log(nb['pNoneTotal'])
    pGood = np.sum(test_review_vector * nb['pGoodVector']) + np.log(nb['pGoodTotal'])
    pBad = np.sum(test_review_vector * nb['pBadVector']) + np.log(nb['pBadTotal'])
    print(pNone, pGood, pBad)
    
    classification = max(pNone, pGood, pBad)
    if (classification == pNone):
        return (test_class == [0, 0])
    elif (classification == pGood):
        return (test_class == [1, 0])
    elif (classification == pBad):
        return (test_class == [0, 1])
    return False
        

# Load the pre-processed dataset
# fix the 522th line
df = pd.read_csv("/home/surajpal/misc/dataset1.1.csv")
#df.to_csv("/home/surajpal/misc/dataset1.1.csv")


# Extract Review texts, and its quality 
# [Good, Bad, None] => [[1, 0]. [0, 1], [0, 0]]
review_text = df['review_text']

good, bad = df['good'], df['bad']
review_quality = []
if (len(good) == len(bad)):
    for i in range(0, len(good)):
        review_quality.append([good[i], bad[i]])

vocabDict = createVocabDict(review_text)

train_review_text, train_review_quality, test_review_text, test_review_quality = \
train_test_split(review_text, review_quality, 0.6)



train_review_text_matrix = []
for i in range(0, int(len(train_review_text))):
    train_review_text_matrix.append(createWordVector(vocabDict, train_review_text[i]))
    
test_review_text_matrix = []
for i in range(18000, 18000 + int(len(test_review_text))):
    print(i)
    test_review_text_matrix.append(createWordVector(vocabDict, test_review_text[i]))
    
nb = train_naive_bayes(train_review_text_matrix, train_review_quality, vocabDict)

errorCount = 0
for i in range(0, len(test_review_text_matrix)):
    first_test = classifier(test_review_text_matrix[i], test_review_quality[i], nb)
    if (first_test == False):
        errorCount += 1
    print(first_test)
print(errorCount)

    
