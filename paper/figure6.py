import sys
from plot import plot_depth_guess

# output file name
figurename='figures/figure6'

# setting the result csv path
resultspath = '../results/allresults.csv'

datasets = ['compas', 'broward_general_2y', 'fico', 'spiral']

# do the plots
for dataset in datasets:
    plot_depth_guess(figurename, resultspath, dataset)


# import sys
# from plot import plot_depth_guess

# # output file name
# figurename='figures/figure6'

# # setting the excel path
# excelpath = '../results/combined_data.xlsx'

# # yaxis metrics
# metrics = ['Training Time', 'Training Accuracy', 'Test Accuracy']

# # the datasets
# dataset='2y_score'

# # algorithsm to plot
# regularization = 0.0005
# plot_depth_guess(figurename, excelpath, dataset=dataset, reg=regularization)
