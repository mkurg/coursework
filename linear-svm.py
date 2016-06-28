#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn import svm
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report

loc_train = "data.csv/data_lem_tags_train_bow.csv"
loc_test = "data.csv/data_lem_tags_test_bow.csv"
loc_submission = "data.csv/kaggle.forest.submission.csv"

df_train = pd.read_csv(loc_train, header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.bool, "conversion1":pd.np.bool, "conversion2":pd.np.bool, "conversion3":pd.np.bool, "tags":pd.np.object, "acm":pd.np.object, "action":pd.np.object, "agregator":pd.np.object, "amn":pd.np.object, "city":pd.np.object, "contacts":pd.np.object, "date":pd.np.object, "geo":pd.np.object, "poi":pd.np.object, "price":pd.np.object, "stars":pd.np.object, "time":pd.np.object, "wh":pd.np.object, "other":pd.np.object, "bag_of_words":pd.np.bool})
df_test = pd.read_csv(loc_test, header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.object, "conversion1":pd.np.bool, "conversion2":pd.np.object, "conversion3":pd.np.object, "tags":pd.np.object, "acm":pd.np.object, "action":pd.np.object, "agregator":pd.np.object, "amn":pd.np.object, "city":pd.np.object, "contacts":pd.np.object, "date":pd.np.object, "geo":pd.np.object, "poi":pd.np.object, "price":pd.np.object, "stars":pd.np.object, "time":pd.np.object, "wh":pd.np.object, "other":pd.np.object, "bag_of_words":pd.np.bool}, na_values=["False"])

print(df_train.dtypes)
print(df_test.dtypes)

#feature_cols = [col for col in df_train.columns if col not in ['lemmatized','id', 'conversion', 'tags', 'city', 'other']]
feature_cols = [col for col in df_train.columns if col not in ['lemmatized','id', 'conversion', 'conversion1','conversion2', 'conversion3', 'tags']]
print(feature_cols)

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

clf = svm.LinearSVC(max_iter=3000, verbose=True)
clf.fit(X_train, y)

df_test = pd.read_csv(loc_test, header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.object, "conversion1":pd.np.bool, "conversion2":pd.np.object, "conversion3":pd.np.object, "tags":pd.np.object, "acm":pd.np.object, "action":pd.np.object, "agregator":pd.np.object, "amn":pd.np.object, "city":pd.np.object, "contacts":pd.np.object, "date":pd.np.object, "geo":pd.np.object, "poi":pd.np.object, "price":pd.np.object, "stars":pd.np.object, "time":pd.np.object, "wh":pd.np.object, "other":pd.np.object, "bag_of_words":pd.np.bool}, na_values=["False"])

feature_cols = [col for col in df_train.columns]
df_test = MultiColumnLabelEncoder(columns = feature_cols).fit_transform(df_test)

z = df_test['conversion']

predictions = list(clf.predict(X_test))

with open(loc_submission, "wb") as outfile:
	outfile.write("id,conversion,real\n")
	for e, val in enumerate(predictions):
		outfile.write("%s,%s,%s\n"%(test_ids[e],val,z[e]))

print(clf.score(X_test, z))
#print(clf.feature_importances_)
print(feature_cols)

target_names = ['worked', 'not worked']

print(classification_report(z, predictions, target_names=target_names))