import numpy as np
import csv
import os
import pathlib
import platform
from datetime import datetime
import hashlib
import json
from resource import getrusage, RUSAGE_SELF
import time

import numpy as np
import pandas as pd
import time
#from model.cart import CART
from model.corels import CORELS
from model.dl85 import DL85
#from model.gosdt import GOSDT
#from model.osdt import OSDT
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import KFold, cross_val_score


# runs a single configuration on the loaded dataset, with the supplied configuration
def do_run_config(X_train, y_train, X_test, y_test, path_to_labels, cfg) :
    print("preparing algorithm.", flush=True)
    # now create the models
    if cfg['algorithm'] == "dl85" or cfg['algorithm'] == "dl8.5" :
        model = DL85(depth=cfg['depth_budget'],  # subtract one to limit the depth
                     regularization = cfg['regularization'],
                     time_limit=cfg['time_limit'],
                     preprocessor="none",
                     warm = None if path_to_labels is None else np.genfromtxt(path_to_labels, skip_header=1)) #the warm labels for dl8.5 require an array, not a file pointer

    elif cfg['algorithm'] == "dl85_no_min_sup" or cfg['algorithm'] == "dl8.5_no_min_sup":
        model = DL85(depth=cfg['depth_budget'],
                     time_limit=cfg['time_limit'],
                     preprocessor="none",
                     warm = None if path_to_labels is None else np.genfromtxt(path_to_labels, skip_header=1)) #the warm labels for dl8.5 require an array, not a file pointer

    elif cfg['algorithm'] == "osdt":
        config = {
            "regularization": cfg['regularization'],
            "time_limit": cfg['time_limit']
        }
        model = OSDT(config, preprocessor="complete")

    elif cfg['algorithm'] == "gosdt":
        config = {
            "regularization": cfg['regularization'],
            "similar_support": False,
            "strong_indifference": False,
            "time_limit": cfg['time_limit'],
            "depth_budget": cfg['depth_budget']
        }
        if cfg['lb_guess']:
            config['warm_LB'] = True
            config['path_to_labels'] = path_to_labels
        model = GOSDT(config)

    elif cfg['algorithm'] == "gosdt_count_subproblems":
        config = {
            "regularization": cfg['regularization'],
            "similar_support": False,
            "strong_indifference": False,
            "time_limit": cfg['time_limit'],
            "depth_budget": cfg['depth_budget']
        }
        model = GOSDT(config)

    elif cfg['algorithm'] == "corels":
        model = CORELS(regularization=cfg['regularization'])
    elif cfg['algorithm'] == "cart":
        model = CART(depth=cfg['depth_budget'], regularization=cfg['regularization'])
    else :
        print("unrecognized algorithm: {}".format(cfg['algorithm']))
        raise Exception


    # ok now train the model, may throw an exception
    print("training model", flush=True)
    model.fit(X_train, y_train)

    print("evaluate the model, extracting tree and scores", flush=True)

    # store the results
    cfg["train_acc"] = model.score(X_train, y_train)
    cfg["train_err"] = model.error(X_train, y_train)
    cfg["test_acc"] = model.score(X_test, y_test)
    cfg["test_err"] = model.error(X_test, y_test)
    cfg["depth"] = model.max_depth()
    cfg["leaves"] = model.leaves()
    cfg["nodes"] = model.nodes()
    cfg["time"] = model.utime
    cfg['systime'] = model.stime
    cfg["lb"] = model.lb
    cfg["ub"] = model.ub
    cfg["loss"] = model.reported_loss

    return cfg, model.json()


# returns the basename to store the results, we just return a hash here
def confighash(cfg) :
    # don't take the idx and optimal depth as part of the hash
    idx = cfg['idx']
    optdepth = cfg['optimaldepth']
    cfg['idx'] = 0
    cfg['optimaldepth'] = 0
    hash =  hashlib.sha256(json.dumps(cfg, indent=2, sort_keys=True).encode('utf-8')).hexdigest()
    cfg['idx'] = idx
    cfg['optimaldepth'] = optdepth
    cfg['hash'] = hash
    return hash

    # common = "{}_{}_{}_{}_{}".format(cfg['dataset'], cfg['algorithm'], cfg['regularization'], cfg['depth_budget'], cfg['fold'])
    # if cfg["thresh"] :
    #     common = "{}_{}-{}-{}".format(common, "thresh", cfg['n_est'], cfg['max_depth'])

    # if cfg["lb_guess"]:
    #     common = "{}_{}-{}-{}".format(common, "lbguess", cfg['lb_n_est'], cfg['lb_max_depth'])

    # return common

# fit the tree using gradient boosted classifier
def fit_boosted_tree(X, y, n_est=10, lr=0.1, d=1):
    clf = GradientBoostingClassifier(loss='deviance', learning_rate=lr, n_estimators=n_est, max_depth=d,
                                    random_state=42)
    clf.fit(X, y)
    out = clf.score(X, y)
    scores = cross_val_score(clf, X, y, cv=5)
    return clf, out, scores


# perform cut on the dataset
def cut(X, ts):
    df = X.copy()
    colnames = X.columns
    for j in range(len(ts)):
        for s in range(len(ts[j])):
            X[colnames[j]+'<='+str(ts[j][s])] = 1
            k = df[colnames[j]] > ts[j][s]
            X.loc[k, colnames[j]+'<='+str(ts[j][s])] = 0
        X = X.drop(colnames[j], axis=1)
    return X


# compute the thresholds
def get_thresholds(X, y, n_est, lr, d, backselect=True):
    # got a complaint here...
    y = np.ravel(y)
    # X is a dataframe
    clf, out, score = fit_boosted_tree(X, y, n_est, lr, d)
    #print('acc:', out, 'acc cv:', score.mean())
    thresholds = []
    for j in range(X.shape[1]):
        tj = np.array([])
        for i in range(len(clf.estimators_)):
            f = clf.estimators_[i,0].tree_.feature
            t = clf.estimators_[i,0].tree_.threshold
            tj = np.append(tj, t[f==j])
        tj = np.unique(tj)
        thresholds.append(tj.tolist())

    X_new = cut(X, thresholds)
    clf1, out1, scorep = fit_boosted_tree(X_new, y, n_est, lr, d)
    #print('acc','1:', out1, 'acc1 cv:', scorep.mean())

    outp = 1
    Xp = X_new.copy()
    clfp = clf1
    itr=0
    if backselect:
        while outp >= out1 and itr < X_new.shape[1]-1:
            vi = clfp.feature_importances_
            if vi.size > 0:
                c = Xp.columns
                i = np.argmin(vi)
                Xp = Xp.drop(c[i], axis=1)
                clfp, outp, scorep = fit_boosted_tree(Xp, y, n_est, lr, d)
                itr += 1
            else:
                break
        Xp[c[i]] = X_new[c[i]]
        _, _, scorep = fit_boosted_tree(Xp, y, n_est, lr, d)

    h = Xp.columns
    #print('features:', h)
    return Xp, thresholds, h, scorep

# compute the thresholds
def compute_thresholds(X, y, cfg) :
    # set LR to 0.1
    lr = 0.1
    start = time.perf_counter()
    X, thresholds, header, score = get_thresholds(X, y, cfg['n_est'], lr, cfg['max_depth'], backselect=True)
    guess_time = time.perf_counter()-start

    # store the guess time
    cfg['guesstime'] = guess_time
    cfg['guess_acc'] = score.mean()
    return X, thresholds, header, cfg

def load_dataset(file) :
    tstart = time.perf_counter()
    datasetfile = file.with_suffix(file.suffix + ".feather")
    if datasetfile.exists() :
        df = pd.read_feather(datasetfile)
    else :
        datasetfile = file.with_suffix(file.suffix + ".csv.gz")
        df = pd.read_csv(datasetfile, sep=";")
    tend = time.perf_counter()
    print(f"dataset file: {datasetfile}")
    print(f"duration loading: {tend - tstart}s", flush=True)
    return df



def get_thresh_dataset(cfg, datasetdir, lb_guess = False) :
    print("dataset: getting threshold dataset {}".format(cfg['dataset']))

    df = load_dataset(datasetdir / "original_datasets" / cfg["dataset"] / cfg["dataset"])

    X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
    h = df.columns[:-1]

    if cfg['fold'] is not None:
        # 5-fold cross validation
        kf = KFold(n_splits=5, shuffle=True, random_state=2021)
        indices = list(kf.split(X))
        (train_index, test_index) = indices[cfg['fold']]

        X_train = pd.DataFrame(X[train_index], columns=h)
        y_train =  pd.DataFrame(y[train_index])
        X_test = pd.DataFrame(X[test_index], columns=h)
        y_test = pd.DataFrame(y[test_index])

        X_train, thresholds, header, cfg = compute_thresholds(X_train, y_train, cfg)

        # get test dataset
        X_test = cut(X_test, thresholds)
        X_test = X_test[header]
    else :
        X = pd.DataFrame(X, columns=h)
        X_train, thresholds, header, cfg = compute_thresholds(X, y, cfg)
        y_train = pd.DataFrame(y)
        X_test = X_train
        y_test = y_train

        train_index = np.arange(0, X_train.shape[0])
        test_index = np.arange(0, X_train.shape[0])

    return X_train, y_train, X_test, y_test, cfg



def get_non_thresh_dataset(cfg, datasetdir) :
    print("dataset: getting non threshold dataset {}".format(cfg['dataset']))
    if cfg['fold'] is not None:
        testfile = datasetdir / "binarized_datasets" / cfg['dataset'] / '{}_binarized_datasets.test{}'.format(cfg['dataset'], cfg['fold'])
        trainfile = datasetdir / "binarized_datasets" / cfg['dataset'] / '{}_binarized_datasets.train{}'.format(cfg['dataset'], cfg['fold'])
        print(f"dataset: testfile {testfile}")
        print(f"dataset: trainfile {trainfile}")
        train_csv = load_dataset(trainfile)
        test_csv = load_dataset(testfile)
        X_train  = pd.DataFrame(train_csv.iloc[:,:-1])
        X_test = pd.DataFrame( test_csv.iloc[:,:-1])
        y_train = pd.DataFrame(train_csv.iloc[:,-1])
        y_test = pd.DataFrame(test_csv.iloc[:,-1])
    else :
        datasetfile = datasetdir / "binarized_datasets" / cfg['dataset'] / '{}_binarized_datasets'.format(cfg['dataset'])
        print(f"dataset: {datasetfile}")
        train_csv = load_dataset(datasetfile)
        X_train = pd.DataFrame(train_csv.iloc[:,:-1])
        y_train = pd.DataFrame(train_csv.iloc[:,-1])
        X_test = X_train
        y_test = y_train

    return X_train, y_train, X_test, y_test

def get_lb_guess_dataset_with_labels(cfg, datasetdir, correctness = False, thresh = True) :
    if thresh:
        #in thresh case, we currently train the warm lb GBDT on the thresholded dataset.
        X_train, y_train, X_test, y_test, cfg = get_thresh_dataset(cfg, datasetdir)
    else:
        #in non-thresh case, we train on the original, not the binarized, dataset (faster training time, same space of models considered)
        #TODO: decide whether we need to include the time to load the original dataset in our reported time for warm_lb
        df = load_dataset(datasetdir / "original_datasets" / cfg["dataset"] / cfg["dataset"])

        X, y = df.iloc[:,:-1].values, df.iloc[:,-1].values
        h = df.columns[:-1]

        if cfg['fold'] is not None:
            # 5-fold cross validation
            kf = KFold(n_splits=5, shuffle=True, random_state=2021)
            indices = list(kf.split(X))
            (train_index, test_index) = indices[cfg['fold']]

            X_train = pd.DataFrame(X[train_index], columns=h)
            y_train =  pd.DataFrame(y[train_index])
            X_test = pd.DataFrame(X[test_index], columns=h)
            y_test = pd.DataFrame(y[test_index])

        else :
            X_train = pd.DataFrame(X, columns=h)
            y_train = pd.DataFrame(y)
            X_test = X_train
            y_test = y_train

    print("dataset: getting warm lower-bound labels for {}".format(cfg['dataset']))

    start_time = time.perf_counter()
    #make warm labels (predictions of GBDT for the training set - TODO: handle multiclass)
    clf = GradientBoostingClassifier(n_estimators=cfg['lb_n_est'], max_depth=cfg['lb_max_depth'], random_state=42)
    clf.fit(X_train, y_train.values.flatten())
    warm_labels = clf.predict(X_train)
    if correctness:
        # use alternative representation of warm_labels if required (1 means correct prediction, 0 means incorrect)
        correct_preds = np.zeros(len(warm_labels))
        correct_preds[warm_labels == y_train.values.flatten()] = 1
        warm_labels = correct_preds
    elapsed_time = time.perf_counter() - start_time

    cfg['lb_time'] = elapsed_time

    print(f"dataset warm lb: {elapsed_time}")

    test_preds = clf.predict(X_test)
    cfg['lb_test_acc'] = np.mean(test_preds == y_test.values.flatten())
    cfg['lb_test_err'] = 1 - cfg['lb_test_acc']
    cfg['lb_train_acc'] = np.mean(clf.predict(X_train) == y_train.values.flatten())
    cfg['lb_train_err'] = 1 - cfg['lb_train_acc']


    # save the labels as a tmp file and return the path to it.
    labelsdir = pathlib.Path('/tmp/warm_lb_labels')
    labelsdir.mkdir(exist_ok=True, parents=True)

    cfghash = cfg['hash']
    labelpath = labelsdir / 'treebench-{}.tmp'.format(cfghash)
    pd.DataFrame(warm_labels).to_csv(labelpath, header="class_labels",index=None) # TODO: verify this formats correctly for gosdt (shouldn't require headers)

    #if thresh is false, we haven't yet loaded the train and test sets to be passed to gosdt. We do so now
    # (ASSUMES THE ORDER OF THE DATA POINTS IS THE SAME IN THE ORIGINAL AND BINARIZED DATASETS)
    if not thresh:
        X_train, y_train, X_test, y_test = get_non_thresh_dataset(cfg, datasetdir)

    return X_train, y_train, X_test, y_test, str(labelpath), cfg

def asbool(val : str):
    if val is None or val == "None":
        return False
    if str(val).lower() == "true" :
        return True
    elif str(val).lower() == 'false' :
        return False
    print(f"WARNING: calling default bool() function on {val}")
    return bool(val)

def asint(val):
    if val == "None" or val is None or val == "" :
        return None
    else:
        return int(val)

def update_errorfile(error_file_path, cfg) :

    try :
        error_file = open(error_file_path, "w")
    except Exception as e:
        print("ERROR: could not open file {} for writing".format(error_file_path))
        raise e

    # write the error file, this is to find the runs that caused an error
    error_file_writer = csv.DictWriter(error_file, fieldnames=cfg.keys())
    error_file_writer.writeheader()
    error_file_writer.writerow(cfg)
    error_file.close()

# run a single configuration
def run_single_configuration(cfg, force = False) :

    tstart = time.perf_counter()

    # ---------------------------------------------------------------------------------------------
    # CONFIG SETUP
    # ---------------------------------------------------------------------------------------------

    # default configuration values for fields not present int `cfg`
    default_cfg = {
        # the index of the configuration
        "idx" : (asint, 0),

        # DATASET SETTINGS

        # the used dataset
        "dataset"  : (str, 'compass'),
        # whether or not we used threshold guessing
        "thresh" : (asbool, False),
        # whether or not we used lowerbound guesses
        "lb_guess" : (asbool, False),
        # which fold of the dataset was used
        "fold" : (asint, 0),

        # ALGORITHM SETTINGS

        # the used algorithm
        "algorithm" : (str, "gosdt"),
        # the used regularization
        "regularization": (float, 0.001),
        # the depth budget
        "depth_budget" : (asint, 0),
        # the known optimal depth for this configuration
        "optimaldepth" : (asint, 0),
        # the timelimit to for running the algorithm
        "time_limit" : (asint, 1800),

        # HYPER PARAMETERS FOR GUESSING

        # the maximum depth used for creating the thresholded dataset
        "max_depth" : (asint, 2),
        # the number of estimators for for creating the thresholded dataset
        "n_est" : (asint, 100),
        # the maximum depth used for the warm lb labels
        "lb_max_depth" : (asint, 2),
        # the number of estimators for the warm lb labels
        "lb_n_est" : (asint, 100),
        # did we use cross validation
        "cross_validate" : (asbool, True),
    }
        # RESULTS
    results_template = {
        # the host name on which the run took place
        "hostname" : platform.node(),
        # timestamp when it ran
        "datetime" : datetime.now().strftime("%Y%m%d-%H%M%S"),

        # the cpu information
        "cpu" : "unknown",

        # the versions for tree benchmark and gosdt
        "gosdtversion" : 'unknown',
        "treebenchmarkversion" : 'unknown',

        # time to load the dataset
        "dateset_loading" : -1,

        # subsamples of the dataset
        "subsamples" : None,
        # the binary sub features
        "binsubfeatures" : None,
        # training accuracy
        "train_acc" : -1,
        # training error
        "train_err" : -1,
        # test accuracy
        "test_acc" : -1,
        # test error
        "test_err" : -1,
        # the depth of the tree
        "depth" : -1,
        # the number of leaves
        "leaves" : -1,
        # the number of nodes
        "nodes"  : -1,
        # the runtime
        "time" : -1,
        # system time
        "systime" : -1,
        # the maximum memory usage
        "mem_mb" : -1,
        # time to perform the  guessing
        "guesstime" : -1,
        # accurancy of the guessing
        "guess_acc" : -1,
        # time to generate the lables
        "lb_time" : -1,
        # train accuracy from the reference model generating the labels
        "lb_train_acc": -1,
        # train error from the reference model generating the labels
        "lb_train_err": -1,
        # test accuracy from the reference model generating the labels
        "lb_test_acc": -1,
        # train error from the reference model generating the labels
        "lb_test_err": -1,
        # the lower bound of the tree
        "lb" : -1,
        # the upperbound of the tree
        "ub" : -1,
        # the loss of the tree
        "loss" : -1,
        # the status of the run
        "status" : "PENDING"
    }

    for k in default_cfg :
        (cast, val) = default_cfg[k]
        if k not in cfg:
            print("NOTICE: config - using default value for field field '{}' = '{}'".format(k, val))
            cfg[k] = val
        else :
            cfg[k] = cast(cfg[k])

    for k in cfg :
        if k not in default_cfg :
            print(f"NOTICE: config - field '{k}' part of cfg, but not default_cfg")

    print("############################################################", flush = True)
    print("# {} - {} - {} - {} - {}".format(cfg['dataset'], cfg['algorithm'], cfg['fold'], cfg['depth_budget'], cfg['regularization']))
    print("############################################################", flush = True)

    # dump the configuration
    print(json.dumps(cfg, indent=2,sort_keys=True))

    # ---------------------------------------------------------------------------------------------
    # RESULT FILES
    # ---------------------------------------------------------------------------------------------

    # Set the results file names
    resultsdirroot = pathlib.Path("results")

    # get the configuration hash
    cfghash = confighash(cfg)
    print(f"config_hash: {cfghash}")
    print(f"config_hash: {cfg['hash']}")
    print(f"config_id: {cfg['idx']}")

    # get the results directory
    resultsdir = resultsdirroot / "runlogs" / cfghash[0:2] / cfghash

    # create the results directory
    resultsdir.mkdir(parents = True, exist_ok= True)

    # the actual output files
    result_file_path = resultsdir /  "results.csv"
    error_file_path = resultsdir /  "error.csv"
    temp_file_path = resultsdir / "results.tmp"
    tree_file_path = resultsdir / "tree.json"
    cfg_file_path = resultsdir / "config.json"

    # check if we've already run this configuration
    if result_file_path.exists() and not force:
        print("already run. (success) skipping...")
        return # This set of trials is already complete

    # is there a temp file, that indicates that we may have had an error
    if temp_file_path.exists() and not force:
        print("already run. (with failure) skipping...")
        return # This set of trials is already complete

    # get the CPU information
    for line in os.popen('lscpu').read().splitlines() :
        if not line.startswith('Model name') :
            continue
        results_template['cpu'] = line.split(':')[1].strip()
        break

    # store the versions
    if 'TB_ENV_GOSDT_VERSION' in os.environ :
        results_template['gosdtversion'] = os.environ['TB_ENV_GOSDT_VERSION']
    else :
        print("NOTE: environment TB_ENV_GOSDT_VERSION is not set!")

    if 'TB_ENV_TREE_BENCHMARK_VERSION' in os.environ :
        results_template['treebenchmarkversion'] = os.environ['TB_ENV_TREE_BENCHMARK_VERSION']
    else :
        print("NOTE: environment TB_ENV_TREE_BENCHMARK_VERSION is not set!")

    # write the configuration map
    configmapdir = resultsdirroot / "configmap"
    configmapdir.mkdir(parents = True, exist_ok= True)
    configmap = configmapdir / str(cfg['idx'])
    with open(configmap, 'w') as f:
        f.write(str(cfghash))
        f.close()

    # store the configuration json
    with open(cfg_file_path, 'w') as f:
        json.dump(cfg, f, indent=2)

    # now copy the results fields
    for k in results_template :
        if k in cfg:
            print("WARNING: field '{}' already in the cfg file???".format(k))
        cfg[k] = results_template[k]


    # ---------------------------------------------------------------------------------------------
    # Open the result files
    # ---------------------------------------------------------------------------------------------

    # try to create the json file to store the tree
    try:
        tree_file = open(tree_file_path, "w")
    except Exception as e:
        print("ERROR: could not open file {} for writing".format(tree_file_path))
        raise e

    # create the CSV file for writing the results
    try :
        temp_file = open(temp_file_path, "w")
    except Exception as e:
        print("ERROR: could not open file {} for writing".format(temp_file_path))
        raise e

    update_errorfile(error_file_path, cfg)

    # write the header to the CSV file
    temp_file_writer = csv.DictWriter(temp_file, fieldnames=cfg.keys())
    temp_file_writer.writeheader()

    # ---------------------------------------------------------------------------------------------
    # DATASET LOADING
    # ---------------------------------------------------------------------------------------------

    tdataset = time.perf_counter()
    print(f"duration preparations: {tdataset - tstart}s", flush=True)

    # the root directories of the datasets and the results
    datasetdir = pathlib.Path("datasets")

    # the generated labels path
    path_to_labels = None

    # set the current run status
    cfg['status'] = "DATASET_LOADING"

    try :
        # we want to run the lowerbound guess datasets
        if cfg['lb_guess'] :
            # if we're using dl85, we need the warm labels to indicate correctness of the predictions,
            # not the predictions themselves. we handle the algorithm with and without a dot here.
            correctness = cfg['algorithm'].startswith("dl85") or cfg['algorithm'].startswith("dl8.5")

            X_train, y_train, X_test, y_test, path_to_labels, cfg = get_lb_guess_dataset_with_labels(cfg, datasetdir, correctness, cfg['thresh'])

        # we use thresholding, but no warm_lb dataset
        elif cfg['thresh']:
            X_train, y_train, X_test, y_test, cfg = get_thresh_dataset(cfg, datasetdir)

        # no thresholding, running on the binarized datasets
        else :
            X_train, y_train, X_test, y_test = get_non_thresh_dataset(cfg, datasetdir)
    except Exception as e:
        print(f"ERROR: Failed to load dataset: {cfg['dataset']}", flush=True)
        print(str(e))
        cfg['status'] = "DATASET_FAILURE"
        temp_file_writer.writerow(cfg)
        temp_file.close()
        error_file_path.unlink()
        return

    n, p = X_train.shape
    # subsamples of the dataset
    cfg["subsamples"] = n
    # the binary sub features
    cfg["binsubfeatures"] = p

    trun = time.perf_counter()
    print(f"duration dataset: {trun - tdataset}s", flush=True)
    cfg['dateset_loading'] = trun - tdataset

    update_errorfile(error_file_path, cfg)

    # ---------------------------------------------------------------------------------------------
    # Run the configuration
    # ---------------------------------------------------------------------------------------------

    # set the current run status
    cfg['status'] = "RUNNING"

    try :
        result, tree = do_run_config(X_train, y_train, X_test, y_test, path_to_labels, cfg)
    except Exception as e:
        print(f"ERROR: Failed to run algorithm `{cfg['algorithm']}`", flush=True)
        print(str(e))
        cfg['status'] = "RUN_FAILURE"
        temp_file_writer.writerow(cfg)
        temp_file.close()
        error_file_path.unlink()
        return

    # get the resource usage,
    s = getrusage(RUSAGE_SELF)
    # ru_maxrss is at pos 2, and the amount is in KiB
    result['mem_mb'] = s[2] / 1024
    result['status'] = "OK"

    tmodel = time.perf_counter()
    print(f"duration model: {tmodel - tdataset}s", flush=True)

    # ---------------------------------------------------------------------------------------------
    # Save the results
    # ---------------------------------------------------------------------------------------------

    # write the results and close file
    temp_file_writer.writerow(result)
    temp_file.close()

    # write the tree file
    tree_file.write(tree)
    tree_file.close()

    # now rename the fle to mark the results
    os.rename(temp_file_path, result_file_path) # commit this file

    # remove the error file
    error_file_path.unlink()

    # ---------------------------------------------------------------------------------------------
    # Clean up temp files
    # ---------------------------------------------------------------------------------------------

    if path_to_labels is not None:
        pathlib.Path(path_to_labels).unlink()

    tend = time.perf_counter()
    print(f"duration running: {tend - tstart}s", flush=True)
