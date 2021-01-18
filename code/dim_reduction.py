#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Jan 16 10:43:34 2021

@author: caboe
"""

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import cross_val_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.tokenize import word_tokenize
import numpy as np
import matplotlib.pyplot as plt
from nltk.stem import WordNetLemmatizer
from sklearn.decomposition import TruncatedSVD
import os
import pandas as pd

from sklearn.model_selection import cross_val_score, train_test_split

%matplotlib qt

# helper functions ###
# modify/remove to make it my own
def getLemms(s):
    return [wordnet_lemmatizer.lemmatize(t) for t in nltk.tokenize.word_tokenize(s)]

def getDTM(vect,arr):
    return vect.fit_transform(arr)

def DTMtoDF(dtm,vect):
    return pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())

def plotTokens(tokens,t_df):
    plt.scatter(t_df[0], t_df[1])
    for row in t_df.values:
        plt.annotate(s=tokens[int(row[0])],(row[1], row[2]))
    plt.show()
    
def visualizeTokens(t_df,mdl,cnt=None):
    X_T = t_df.T
    tokens = X_T.index.values.tolist()
    z = dr_mdl.fit_transform(X_T)
    z_df = pd.DataFrame(z)
    z_df['sum'] = abs(z_df[0]) + abs(z_df[1])
    z_df = z_df.sort_values('sum',ascending=False)
    z_df = z_df.reset_index().rename(columns={'index':'idx'})
    plotTokens(tokens,z_df.iloc[:cnt])
    return z,tokens

def printVarianceExp(mdl):
    print('Components: {} | Explained Variance: {:.3f}%'.format(len(mdl.components_), mdl.explained_variance_ratio_.sum()*100))

def exploreTokens(vect,arr,mdl,cnt=None):
    t_dtm = getDTM(vect,arr)
    t_df = DTMtoDF(t_dtm,vect)
    if showTokens:
        visualizeTokens(t_df,mdl,cnt)
        

### global variables ####
data_pth = '../data/'
text_fn = 'news.csv'
showTokens = True
wordnet_lemmatizer = WordNetLemmatizer()
random_state = 42
df = pd.read_csv(os.path.join(data_pth,text_fn))

# instantiate vectorizer
X_col = 'title'
y_col = 'label'
vect = CountVectorizer(tokenizer=my_tokenizer,stop_words = 'english')
dr_mdl = TruncatedSVD(random_state=random_state)

exploreTokens(vect, df[X_col], dr_mdl)

#vect = TfidfVectorizer(tokenizer=my_tokenizer, stop_words=stopwords)

# transform titles from columns of single strings to a sparse DTM: document term matrix
dtm = vect.fit_transform(df['title'])

# convert DTM sparse matrix to a sparse DataFrame with TOKENS AS COLUMNS
df_dtm = pd.DataFrame(dtm.toarray(),columns=vect.get_feature_names())

# transpose the Document Term Matrix to a Term-Document Matrix with the TOKENS AS ROWS 
# This is necessary to use the Truncated SVD 
new_X = df_dtm.T 

# get a list of the unique tokens in order to graph them with corresponding transformed values
tokens = new_X.index.values.tolist()

# instantiate dim reduction model # PCA(), FA(), ICA(), TruncatedSVD()
mdl = TruncatedSVD(random_state=42)
# Transform term-document matrix 
Z = mdl.fit_transform(new_X)
# Show the amount of variace explained by these two components 
print('Components: {} | Explained Variance: {:.3f}%'.format(len(mdl.components_), mdl.explained_variance_ratio_.sum()*100))

#attributes
a = mdl.components_ # huh?
# variance of the training samples transformed by a projection to each component
# what?
b = mdl.explained_variance_ 
#Percentage of variance explained by each of the selected components 
c = mdl.explained_variance_ratio_ # this is the
d = mdl.singular_values_

# plot transformed values of each of the tokens
plt.scatter(Z[:,0], Z[:,1])
for i in range(Z.shape[0]):
    plt.annotate(s=tokens[i], xy=(Z[i,0], Z[i,1]))
plt.show()
