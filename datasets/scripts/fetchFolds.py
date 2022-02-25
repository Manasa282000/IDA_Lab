#!/bin/python3

import os
from shutil import copyfile

def fetchFolds(folder_containing_folds, destination, prefix=""):
    folds = (os.listdir(folder_containing_folds))
    for fold in folds: 
        fold_info = fold.split('.')[-2] # The part of the name between the penultimate and final '.' characters 
                                        # contains the fold information (train vs test, fold number)
        name_for_fold = prefix + fold_info + '.csv'
        copyfile(os.path.join(folder_containing_folds, fold), os.path.join(destination, name_for_fold))

    
