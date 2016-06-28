#!/usr/bin/python
# -*- coding: utf-8 -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report
import numpy as np
import pandas as pd  

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = None)
classifier = RandomForestClassifier(n_estimators = 100) 

train = pd.read_csv('data.csv/data_lem_train.csv', header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False)

print train.shape

test = pd.read_csv('data.csv/data_lem_test.csv', header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False)

print test.shape

num_queries = train["lemmatized"].size
clean_train_queries = []

x_train = train['lemmatized']
y_train = train['conversion']

x_test = test['lemmatized'].fillna('')
y_test = test['conversion'].astype('bool')

print "Training...\n"

clf = Pipeline([
    ('vctr', vectorizer),
    ('clf', classifier)
    ])

clf.fit(x_train, y_train)

print test.shape

num_queries = len(test["lemmatized"])
clean_test_queries = []

print "Testing...\n"

y_pred = clf.predict(x_test)
output = pd.DataFrame( data={"id":test["id"], "conversion":y_pred} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )

print classification_report(y_test, y_pred)