import sys
from plot import plot_tradeoff

# output file name
figurename='figures/figure3'

# setting the result csv path
resultspath = '../results/allresults.csv'

# setting the bat tree path
batreepath = '../results/batree_results.csv'

# the datasets
datasets=['compas','broward_general_2y','coupon_carryout','coupon_rest20', 'fico', 'netherlands']

guess_types = ['thresh', 'lb_guess', 'both']

max_depth = {'compas':3, 'broward_general_2y':1, 
        'coupon_carryout':2,'coupon_rest20':2,
        'fico': 1, 'netherlands':2, 'spiral': 3}
n_est = {'compas':20, 'broward_general_2y':40, 
        'coupon_carryout':50,'coupon_rest20':50,
        'fico': 40, 'netherlands':30, 'spiral': 20}
lb_max_depth = {'compas':1, 'broward_general_2y':1, 
        'coupon_carryout':1,'coupon_rest20':1,
        'fico': 1, 'netherlands':1, 'spiral': 1}
lb_n_est = {'compas':100, 'broward_general_2y':100, 
        'coupon_carryout':100,'coupon_rest20':100,
        'fico': 100, 'netherlands':100, 'spiral':100}

# do the plot
for dataset in datasets:
    for guess_type in guess_types:
        # do the plot
        plot_tradeoff(figurename, resultspath, batreepath, dataset, guess_type,
                      max_depth[dataset], n_est[dataset], max_depth[dataset], n_est[dataset])

# import sys
# from plot import plot_tradeoff

# # output file name
# figurename='figures/figure3'

# # setting the excel path
# excelpath = '../results/combined_data.xlsx'
# excelpath2 = '../results/combined_data.xlsx'

# # setting the bat tree path
# batreepath = '../results/batree_experiments.xlsx'

# # the datasets
# dataset='general_2y'

# # algorithsm to plot
# algorithms=['cart', 'dl85', 'gosdt_warm_lb']

# # the depths to plot
# depths = [5, 6, -1]

# # do the plot
# plot_tradeoff(figurename, excelpath, excelpath2, batreepath, dataset=dataset, algorithms=algorithms, depths = depths)
