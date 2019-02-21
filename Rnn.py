# -*- coding: utf-8 -*-
"""
Created on Mon Feb 18 18:57:35 2019

@author: Bhanugoban Nadar
"""

import json
import pandas as pd
import nltk
from nltk import word_tokenize
from nltk.corpus import stopwords
import numpy as np
import random
import re
import string

num_good = 10000
num_bad = 10000
num_neutral = 10000
dataset_src_fn = 'Electronics_5.json'
final_df_name = 'dataset.csv'

col_names = ["reviewer_id", "asin", "review_text", "overall", "category", 
             "good", "bad","Count_Reviews","product_Popularity"]

def read_dataset(fn):
    data = []
    reviewFrequency={}
    productPopularity={}
    with open(fn) as f:
       
        for line in f:
            d = json.loads(line)
            p=reviewFrequency.get(d["reviewerID"],0)
            reviewFrequency[d["reviewerID"]]=p+1
            p=productPopularity.get(d["asin"],0)
            productPopularity[d["asin"]]=p+1
    with open(fn) as f:
        for line in f:
            d = json.loads(line)
            pf, tf = d["helpful"]
            if(tf == 0 ):
                continue
            
            score = (1.0 * pf) / (1.0 * tf)
            row = [d["reviewerID"], 
                   d["asin"], 
                   d["summary"] + ' ' + d["reviewText"],
                   d["overall"],
                   "Electronics",
                   int(score > 0.80),
                   int(score <= 0.20),
                   reviewFrequency[d["reviewerID"]],
                   productPopularity[d["asin"]]
                   ]
            data.append(row)
    return pd.DataFrame(data, columns=col_names)

print ('Creating dataframe...')
df = read_dataset(dataset_src_fn)
df = df.sample(frac=1).reset_index(drop=True) # Randomize entry order
df.head()

df_good = df.loc[df['good'] == 1]
df_good = df_good.sample(frac=1).reset_index(drop=True)
df_good.drop(df_good.index[num_good:], inplace=True)

df_bad = df.loc[df['bad'] == 1]
df_bad = df_bad.sample(frac=1).reset_index(drop=True)
df_bad.drop(df_bad.index[num_bad:], inplace=True)

df_neutral = df.loc[(df['good'] == 0) & (df['bad'] == 0)]
df_neutral = df_neutral.sample(frac=1).reset_index(drop=True)
df_neutral.drop(df_neutral.index[num_neutral:], inplace=True)

print (len(df_good), len(df_bad), len(df_neutral))


df_min = pd.concat([df_good, df_bad, df_neutral], axis=0, join='outer', ignore_index=True)
df_min = df_min.sample(frac=1).reset_index(drop=True)
del df, df_good, df_bad, df_neutral # Free memory
print ("Number of entries:", len(df_min))
print ("Good count:", len(df_min.loc[df_min['good'] == 1]))
print ("Bad count:", len(df_min.loc[df_min['bad'] == 1]))
print ("Neutral count:", len(df_min.loc[(df_min['good'] == 0) & (df_min['bad'] == 0)]))
df_min.head()

stop_words = stopwords.words('english')
word_pattern = re.compile("[A-Za-z]+")
n_entries = len(df_min)
df_norm = pd.DataFrame(columns=col_names, index=range(n_entries))

def normalize_review_text(text):
    def norm_filter(w):
        return w not in stop_words and \
               len(w) > 2
    tokens = nltk.regexp_tokenize(text.lower(), word_pattern)
    return ' '.join(filter(norm_filter, tokens))

for idx in range(n_entries):
    row = df_min.iloc[idx]
    norm_text = normalize_review_text(row['review_text'])
    df_norm.iloc[idx] = [
        row['reviewer_id'],
        row['asin'],
        norm_text,
        row['overall'],
        row['category'],
        row['good'],
        row['bad'],
        row['Count_Reviews'],
        row['product_Popularity']
    ]
    if idx % 10000 == 0 or idx + 1 == n_entries:
        print ('Entry ' + str(idx + 1) + '/' + str(n_entries) + '.')

print ("Finished pre-processing review text.")

df_norm.to_csv(final_df_name)
print ("Saved to disk!")