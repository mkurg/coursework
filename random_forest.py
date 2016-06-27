#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import ensemble
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline

loc_train = "data.csv/data_lem_tags_train.csv"
loc_test = "data.csv/data_lem_tags_test.csv"
loc_submission = "data.csv/kaggle.forest.submission.csv"

df_train = pd.read_csv(loc_train, header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.bool, "tags":pd.np.object, "acm":pd.np.object, "action":pd.np.object, "agregator":pd.np.object, "amn":pd.np.object, "city":pd.np.object, "contacts":pd.np.object, "date":pd.np.object, "geo":pd.np.object, "poi":pd.np.object, "price":pd.np.object, "stars":pd.np.object, "time":pd.np.object, "wh":pd.np.object, "other":pd.np.object})
df_test = pd.read_csv(loc_test, header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.bool, "tags":pd.np.object, "acm":pd.np.object, "action":pd.np.object, "agregator":pd.np.object, "amn":pd.np.object, "city":pd.np.object, "contacts":pd.np.object, "date":pd.np.object, "geo":pd.np.object, "poi":pd.np.object, "price":pd.np.object, "stars":pd.np.object, "time":pd.np.object, "wh":pd.np.object, "other":pd.np.object})

print(df_train.dtypes)
print(df_test.dtypes)

feature_cols = [col for col in df_train.columns if col not in ['lemmatized','id', 'conversion', 'tags', 'city', 'other']]

#le = preprocessing.LabelEncoder()

class MultiColumnLabelEncoder:
    def __init__(self,columns = None):
        self.columns = columns # array of column names to encode

    def fit(self,X,y=None):
        return self # not relevant here

    def transform(self,X):
        '''
        Transforms columns of X specified in self.columns using
        LabelEncoder(). If no columns specified, transforms all
        columns in X.
        '''
        output = X.copy()
        if self.columns is not None:
            for col in self.columns:
                output[col] = LabelEncoder().fit_transform(output[col])
        else:
            for colname,col in output.iteritems():
                output[colname] = LabelEncoder().fit_transform(col)
        return output

    def fit_transform(self,X,y=None):
        return self.fit(X,y).transform(X)

df_train = MultiColumnLabelEncoder(columns = feature_cols).fit_transform(df_train)
df_test = MultiColumnLabelEncoder(columns = feature_cols).fit_transform(df_test)

#df_train = le.fit_transform(df_train)
#df_test = le.fit_transform(df_test)

X_train = df_train[feature_cols]
X_test = df_test[feature_cols]

y = df_train['conversion']
test_ids = df_test['id']

clf = ensemble.RandomForestClassifier(n_estimators=1000, n_jobs=-1, verbose=2)
clf.fit(X_train, y)

z = df_test['conversion']

with open(loc_submission, "wb") as outfile:
	outfile.write("id,conversion\n")
	for e, val in enumerate(list(clf.predict(X_test))):
		outfile.write("%s,%s\n"%(test_ids[e],val))

print(clf.score(X_test, z))
print(clf.feature_importances_)