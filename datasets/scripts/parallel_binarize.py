#!/bin/python3

#
# Expects to be called from the datasets/ folder !!!
#

# TODO: rewrite so that it only does the binarization and not the KFold
#       and move that ability to the makeFolds.py script

import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from joblib import Parallel, delayed
import gc
import multiprocessing
import sys
import csv
import time
import pathlib
pd.options.mode.chained_assignment = None  # default='warn'

CREATE_CSV=False

def binarizejob(df, colnames, tasks) :
    nrows, ncols = df.shape
    dfout = pd.DataFrame(index=np.arange(nrows))
    tot = len(tasks)
    for (tid, (i, v)) in enumerate(tasks):
        if tid % 500 == 0 :
            print(f"{tid} / {tot}")
        dfout[colnames[i]+'<='+str(v)] = True
        k = df[colnames[i]] > v
        dfout[colnames[i]+'<='+str(v)][k] = False

    return dfout


def storejob(dataset, df,  job, fold, idx) :

    datasetrootpath = pathlib.Path('binarized_datasets')
    datasetpath = datasetrootpath / dataset

    # create the data set
    datasetpath.mkdir(exist_ok=True, parents=True)

    X_job = df.iloc[idx].reset_index(drop=True, inplace=False).astype('int8')
    filename = dataset + '_binarized_datasets.' + job + str(fold) + '.feather'
    featherpath = datasetpath / filename
    print(f"Save as feather: {featherpath}")
    print(f"Feather dimensions ", X_job.shape)
    X_job.to_feather(featherpath)

    del X_job
    gc.collect()

    if CREATE_CSV :
        filename = dataset + '_binarized_datasets.' + job + str(fold) + '.csv.gz'
        csvfilepath = datasetpath / filename
        print(f"Save as csv: {csvfilepath}")
        X_job = df.iloc[idx].astype('int8')
        print(f"CSV dimensions ", X_job.shape)
        X_job.to_csv(csvfilepath, sep=";", index=False)
        del X_job
    del df
    gc.collect()


if len(sys.argv) < 2:
    print("Correct usage: python3 binarize.py [name-of-dataset]")

dataset = sys.argv[1]

# read all the chunks
chunksize = 200
dflist = []
for chunk in pd.read_csv('./original_datasets/' + dataset + '/' + dataset + '.csv.gz', chunksize=chunksize, delimiter=";"):
    dflist.append(chunk)

# assemble the data frame
df = pd.concat(dflist)

# how much parallelism we can take
njobs = multiprocessing.cpu_count()
print("parallel binarize with {} jobs".format(njobs))

# workout the column names
colnames = df.columns
tasks = []
for i in range(len(colnames)-1):
    uni = df[colnames[i]].unique()
    uni.sort()
    for v in uni[:-1] :
        tasks.append((i, v))

# calculate the number of items per task
ntasks = len(tasks)
alltasks = [[] for i in range(0, njobs)]
itemspertask = len(tasks) // njobs

# construct the task lists
cnt = 0
for i in range(0, njobs) :
    alltasks[i] = tasks[i*itemspertask:(i+1)*itemspertask]
    cnt += len(alltasks[i])

# append the last ones
alltasks[-1].extend(tasks[njobs * itemspertask:])


# start the binarization jobs
print("starting jobs...")
tstart = time.perf_counter()
results = Parallel(n_jobs=njobs)(delayed(binarizejob)(df, colnames, t) for t in alltasks)

# now concatenate the dataframes again
print("jobs completed, concatenate dataframes")
dfret = pd.concat(results, axis=1, join="inner")

# add the features target column
dfret['target'] = df.iloc[:, -1]

tend = tbinarize = time.perf_counter()

print(f"datasets assembled. time {tend - tstart}")
print(f"original dataset: {df.shape}")
print(f"binarized: {dfret.shape}")

# we can delete the original data frames again
del df
del results
gc.collect()


tstart = time.perf_counter()
print("Creating folds")
kf = KFold(n_splits=5, shuffle=True, random_state=2021)

tasks = []
for j, (train_index, test_index) in enumerate(kf.split(dfret.iloc[:,:-1])) :
    tasks.append(('train', j, train_index))
    tasks.append(('test', j, test_index))

tend = time.perf_counter()
print(f"Folds created. time {tend -tstart}")

print("Saving datasets...")
tstart = time.perf_counter()
Parallel(n_jobs=njobs)(delayed(storejob)(dataset, dfret, job, fold, idx) for (job, fold, idx)  in tasks)
tend = time.perf_counter()
print(f"Datasets saved. time {tend - tstart}")
