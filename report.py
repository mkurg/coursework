#!/usr/bin/python
# -*- coding: utf-8 -*-

import pandas as pd
from sklearn.metrics import classification_report

data = pd.read_csv("data.csv/kaggle.forest.submission.csv", header=0, delimiter=",", doublequote=True, escapechar='\\', quotechar='"', error_bad_lines=False, dtype={"id":pd.np.int32, "conversion":pd.np.bool, "real":pd.np.bool}, na_values=["False"])

y_true = data['real']
y_pred = data['conversion']

target_names = ['worked', 'not worked']

print(classification_report(y_true, y_pred, target_names=target_names))