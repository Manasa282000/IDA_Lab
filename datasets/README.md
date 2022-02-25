# datasets/
The Datasets used for our experiments.

A summary of these datasets and the parameters with which we test them can be 
found in [dataset_info.csv](dataset_info.csv).

Note that all the derived datasets in this folder (binarized, threshold_guess) 
are not version controlled and are instead generated using the 
`generate_derived_datasets.sh` shell script in the `scripts/` folder.

## File Structure: 
```
datasets/
|
+-- dataset_info.csv: Summary of the included datasets and the hyperparameters with which they are trained
+-- original_datasets: The original unedited version of all the datasets we use
     |
     +-- netherlands_source.csv: original netherlands dataset. Source: Nikolaj Tollenaar and PGM Van der Heijden. Which method predicts recidivism best?: a comparison of statistical, machine learning and data mining predictive models. Journal of the Royal Statistical Society: Series A (Statistics in Society), 176(2):565–584, 2013
     +-- coupon_source.csv: original coupon dataset. Source: Tong Wang, Cynthia Rudin, Finale Doshi-Velez, Yimin Liu, Erica Klampfl, and Perry MacNeille. Or’s of and’s for interpretable classification, with application to context-aware recommender systems. arXiv preprint arXiv:1504.07614, 2015
+-- binarized_datasets: The binarized versions of the datasets presented in the original_datasets folder
+-- threshold_guess_datasets: The datasets produced by threshold guessing code found in scripts/create_thresholds.py
+-- scripts/: Contains all of the scripts needed to generate the datasets in this folder.
     |
     +-- fetchFolds.py
     +-- makeFolds.py
     +-- makeThresholds.py
     +-- binarize.py
```

### TODO
1. Adapt/Create the scripts in `scripts/`
2. Obtain/Organize the original datasets
