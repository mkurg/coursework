#!/usr/bin/python
# -*- coding: <encoding name> -*-

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier
import numpy as np
import pandas as pd  

vectorizer = CountVectorizer(analyzer = "word",   \
                             tokenizer = None,    \
                             preprocessor = None, \
                             stop_words = None,   \
                             max_features = None) 

train = pd.read_csv('data.csv/data_lem_train.csv', header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False)

print train.shape

test = pd.read_csv('data.csv/data_lem_test.csv', header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False)

print test.shape

num_queries = train["lemmatized"].size
clean_train_queries = []

print "Cleaning and parsing the training set...\n"

for i in xrange( 0, num_queries ):
    if( (i+1)%1000 == 0 ):
    	print "Query %d of %d\n" % ( i+1, num_queries )       
	clean_train_queries.append( train["lemmatized"][i] )

train_data_features = vectorizer.fit_transform(clean_train_queries)
train_data_features = train_data_features.toarray()

print train_data_features.shape

forest = RandomForestClassifier(n_estimators = 100) 
forest = forest.fit(train_data_features, train["conversion"] )

print test.shape

num_queries = len(test["lemmatized"])
clean_test_queries = []

print "Cleaning and parsing the test set...\n"
for i in xrange(0,num_queries):
	if((i+1) % 1000 == 0):
		print "Query %d of %d\n" % (i+1, num_queries)
	#clean_query = query_to_words(test["lemmatized"][i])
	clean_test_queries.append(test["lemmatized"][i])

test_data_features = vectorizer.transform(clean_test_queries)
test_data_features = test_data_features.toarray()

result = forest.predict(test_data_features)
output = pd.DataFrame( data={"id":test["id"], "conversion":result} )
output.to_csv( "Bag_of_Words_model.csv", index=False, quoting=3 )