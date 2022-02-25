# Generate the config.csv file based on the current state of the ../datasets/dataset_info.csv file
# !!! Must be called from top level tree-benchmark directory.
import pathlib
import csv

timelimit = "1800"

algorithms = ["gosdt", "cart", "dl8.5"]

header_row = [
    "dataset",
    "thresh",
    "lb_guess",
    "fold",
    "algorithm",
    "regularization",
    "depth_budget",
    "optimaldepth",
    "time_limit",
    "max_depth",
    "n_est",
    "lb_max_depth",
    "lb_n_est",
    "cross_validate"
]


def format_row(label, time_limit, algo, dataset, regularization, depth_budget, labels, blackboxparams, fold, thresh,
               optimaldepth=None):
    return [label, time_limit, '', '', '', algo, dataset, regularization, depth_budget, labels, blackboxparams, '', '',
            '', '', '',
            '', '', '', '', '', '', fold, optimaldepth, thresh]


def write_cfg(dataset, thresh, lb_guess, fold, algorithm, regularization, depth_budget, optimaldepth, time_limit,
              max_depth, n_est, lb_max_depth, lb_n_est, cross_validate):
    cfg = [
        # DATASET SETTINGS

        # the used dataset
        dataset,
        # whether or not we used threshold guessing
        thresh,
        # whether or not we used lowerbound guesses
        lb_guess,
        # which fold of the dataset was used
        fold,

        # ALGORITHM SETTINGS

        # the used algorithm
        algorithm,
        # the used regularization
        regularization,
        # the depth budget
        depth_budget,
        # the known optimal depth for this configuration
        optimaldepth,
        # the timelimit to for running the algorithm
        time_limit,

        # HYPER PARAMETERS FOR GUESSING

        # the maximum depth used for creating the thresholded dataset
        max_depth,
        # the number of estimators for for creating the thresholded dataset
        n_est,
        # the maximum depth used for the warm lb labels
        lb_max_depth,
        # the number of estimators for the warm lb labels
        lb_n_est,
        # did we use cross validation
        cross_validate
    ]

    return cfg

#determine depths (handle case where we want to avoid specifying no depth budget)
def depths(row, alg):
    depths_to_try = eval(row["depths (general)"])
    # if we are running dl8.5 or cart, we do not want to provide depth budget 0 (no depth budget) as an argument
    # at least for the current set of experiments (these settings don't always make a lot of sense, and in the case of dl8.5
    # it can make optimization intractable)
    if alg in ("dl8.5", "cart"): 
        depths_to_try.remove(0)
    return depths_to_try

# Handle input
def generate_configs(dataset_info_path, config_info_path):
    with open(dataset_info_path) as dataset_info, open(config_info_path, "a") as config_file:
        print("opened config file successfully")
        info_reader = csv.DictReader(dataset_info, delimiter=',')
        config_writer = csv.writer(config_file, delimiter=',')

        # Write header
        config_writer.writerow(header_row)

        # Write configurations
        for row in info_reader:
            dataset = row["Dataset"]

            #determine folds (handle case where folds is None)
            folds = [None] if row["folds"] in (None, "") else range(0, eval(row["folds"]))

            # Regular run
            for alg in algorithms:  # eval(row["alorithms"]):
                for reg in eval(row["regularizations (general)"]):
                    for depth in depths(row, alg):
                        #loop through all folds (unless there are no folds)
                        for fold in folds:
                            config_writer.writerow(
                                write_cfg(
                                    dataset,
                                    thresh=False,
                                    lb_guess=False,
                                    fold=fold,
                                    algorithm=alg,
                                    regularization=reg,
                                    depth_budget=depth,
                                    optimaldepth=None,
                                    time_limit=timelimit,
                                    max_depth=None,
                                    n_est=None,
                                    lb_max_depth=None,
                                    lb_n_est=None,
                                    cross_validate=False
                                )
                            )

            # Warm_lb
            for alg in algorithms:  # eval(row["alorithms"]):
                #don't run warm_lb tests if alg is cart
                if alg == "cart": 
                    continue
                for reg in eval(row["regularizations (general)"]):
                    for depth in depths(row, alg):
                        for fold in folds:
                            for (lb_n_est, lb_max_depth) in eval(row["reference_labels_paired_n_est_max_depth"]):
                                config_writer.writerow(
                                    write_cfg(
                                        dataset,
                                        thresh=False,
                                        lb_guess=True,
                                        fold=fold,
                                        algorithm=alg,
                                        regularization=reg,
                                        depth_budget=depth,
                                        optimaldepth=None,
                                        time_limit=timelimit,
                                        max_depth=None,
                                        n_est=None,
                                        lb_max_depth=lb_max_depth,
                                        lb_n_est=lb_n_est,
                                        cross_validate=False
                                    )
                                )

            # The thresh on case
            for alg in algorithms:  # eval(row["alorithms"]):
                for reg in eval(row["regularizations (general)"]):
                    for depth in depths(row, alg):
                        for fold in folds:
                            for (n_est, max_depth) in eval(row["threshold_guess_paired_n_est_max_depth"]):
                                config_writer.writerow(
                                        write_cfg(
                                            dataset,
                                            thresh=True,
                                            lb_guess=False,
                                            fold=fold,
                                            algorithm=alg,
                                            regularization=reg,
                                            depth_budget=depth,
                                            optimaldepth=None,
                                            time_limit=timelimit,
                                            max_depth=max_depth,
                                            n_est=n_est,
                                            lb_max_depth=None,
                                            lb_n_est=None,
                                            cross_validate=False
                                        )
                                    )

            # Warm_lb
            for alg in algorithms:  # eval(row["alorithms"]):
                #don't run warm_lb tests if alg is cart
                if alg == "cart": 
                    continue
                for reg in eval(row["regularizations (general)"]):
                    for depth in depths(row, alg):
                        for fold in folds:
                            for (lb_n_est, lb_max_depth) in eval(row["reference_labels_paired_n_est_max_depth"]):
                                for (n_est, max_depth) in eval(row["threshold_guess_paired_n_est_max_depth"]):
                                    config_writer.writerow(
                                        write_cfg(
                                            dataset,
                                            thresh=True,
                                            lb_guess=True,
                                            fold=fold,
                                            algorithm=alg,
                                            regularization=reg,
                                            depth_budget=depth,
                                            optimaldepth=None,
                                            time_limit=timelimit,
                                            max_depth=max_depth,
                                            n_est=n_est,
                                            lb_max_depth=lb_max_depth,
                                            lb_n_est=lb_n_est,
                                            cross_validate=False
                                        )
                                    )

            # optimal depths with thresh on
            for ((n_est, max_depth), reg, optimal_depth) in eval(row["paired_n_est_max_depth_regularizations_optimal_depths (depth_guesses)"]):
                #determine fold (handle case where folds is None)
                fold = None if row["folds"] in (None, "") else 0
                for i in [-2, -1, 0, 1, 2]:
                    config_writer.writerow(
                        write_cfg(
                            dataset=dataset,
                            thresh=True,
                            lb_guess=False,
                            fold=fold,
                            algorithm="gosdt",
                            regularization=reg,
                            depth_budget=optimal_depth + i,
                            optimaldepth=optimal_depth,
                            time_limit=timelimit,
                            max_depth=max_depth,
                            n_est=n_est,
                            lb_max_depth=None,
                            lb_n_est=None,
                            cross_validate=False
                        )
                    )
                        
        print("config file written successfully")

import argparse

if __name__=="__main__":

    parser  = argparse.ArgumentParser()
    parser.add_argument("dataset_info_path", help="path to dataset_info.csv file to be parsed")
    parser.add_argument("config_path", help="path to config file to be generated")
    args = parser.parse_args()

    config_info_path = pathlib.Path(args.config_path)
    dataset_info_path = pathlib.Path(args.dataset_info_path)

    generate_configs(dataset_info_path, config_info_path)
