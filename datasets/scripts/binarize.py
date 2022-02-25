#!/bin/python3

#
# Expects to be called from the datasets/ folder !!!
#

# TODO: rewrite so that it only does the binarization and not the KFold
#       and move that ability to the makeFolds.py script

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import sys
pd.options.mode.chained_assignment = None  # default='warn'

def binarize(df):
    colnames = df.columns
    for i in range(len(colnames)-1):
        uni = df[colnames[i]].unique()
        uni.sort()
        for j in range(len(uni)-1):
            df[colnames[i]+'<='+str(uni[j])] = 1
            k = df[colnames[i]] > uni[j]
            df[colnames[i]+'<='+str(uni[j])][k] = 0
        df = df.drop(colnames[i], axis=1)
    return df

if len(sys.argv) < 2:
    print("Correct usage: python3 binarize.py [name-of-dataset]")

dataset = sys.argv[1]

chunksize = 200
dflist = []
for chunk in pd.read_csv('./original_datasets/' + dataset + '/' + dataset + '.csv', chunksize=chunksize, delimiter=";"):
    dflist.append(chunk)

df = binarize(pd.concat(dflist))
print("Dataframe: ", df.shape)

h = df.columns
h = h[1:]
X = df.iloc[:,1:].values
y = df.iloc[:,0].values
X = X.astype('int32')
y = y.astype('int32')

kf = KFold(n_splits=5, shuffle=True, random_state=2021)
for j, (train_index, test_index) in enumerate(kf.split(X)):
    X_train, X_test = X[train_index], X[test_index]
    y_train, y_test = y[train_index], y[test_index]
    train, test = pd.DataFrame(X_train, columns=h), pd.DataFrame(X_test, columns=h)
    train['target'] = y_train
    test['target'] = y_test
    train.to_csv('./binarized_datasets_seq/' + dataset + '/' + dataset + '_binarized_datasets.train' + str(j) + '.csv.gz', sep=";", index=False)
    test.to_csv('./binarized_datasets_seq/' + dataset + '/' + dataset + '_binarized_datasets.test' + str(j) + '.csv.gz', sep=";", index=False)
    #train.to_csv('./binarized_datasets/' +  'dataGeneral/dataGeneral_binarized_datasets.train'+str(j)+'.csv', sep=";", index=False)
    #test.to_csv('./binarized_datasets/dataGeneral/dataGeneral_binarized_datasets.test'+str(j)+'.csv', sep=";", index=False)
