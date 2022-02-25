import sys
from plot import plot_depth

# output file name
figurename='figures/figure5'

# setting the result csv path
resultspath = '../results/depthresults.csv'

datasets = ['compas']#, 'broward_general_2y', 'fico', 'spiral']

# do the plots
for dataset in datasets:
    plot_depth(figurename, resultspath, dataset, max_depth=3, n_est=20)