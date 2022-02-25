import sys
from plot import plot_runtime

# output file name
figurename='figures/figure2'

# setting the result csv path
resultspath = '../results/allresults.csv'

# the datasets
datasets=['compas', 'broward_general_2y', 'coupon_carryout', 'coupon_rest20', 'fico', 'netherlands']

# the depths to plot
depth_budgets = [4, 6, 0]

# regularization
regs = {'compas':[0.001], 'broward_general_2y':[0.005], 
        'coupon_carryout':[0.001],'coupon_rest20':[0.001],
        'fico': [0.0005], 'netherlands':[0.001], 'spiral': [0.01]}

# max_depth and n_est for thresh guess
max_depth = {'compas':3, 'broward_general_2y':1, 
        'coupon_carryout':2,'coupon_rest20':2,
        'fico': 1, 'netherlands':2, 'spiral': 3}
n_est = {'compas':20, 'broward_general_2y':40, 
        'coupon_carryout':50,'coupon_rest20':50,
        'fico': 40, 'netherlands':30, 'spiral': 20}

# lb_max_depth and lb_n_est for lb guess
lb_max_depth = {'compas':1, 'broward_general_2y':1, 
        'coupon_carryout':1,'coupon_rest20':1,
        'fico': 1, 'netherlands':1, 'spiral': 1}
lb_n_est = {'compas':100, 'broward_general_2y':100, 
        'coupon_carryout':100,'coupon_rest20':100,
        'fico': 100, 'netherlands':100, 'spiral': 100}

# do the plot
for dataset in datasets:
    plot_runtime(figurename, resultspath, dataset, depth_budgets, regs[dataset],
                  max_depth[dataset], n_est[dataset], max_depth[dataset], n_est[dataset])


# import sys
# from plot import plot_runtime

# # output file name
# figurename='figures/figure2'

# # setting the excel path
# excelpath_guess = '../results/combined_data.xlsx'
# excelpath_no_guess = '../results/combined_data.xlsx'

# # the datasets
# dataset='2y_score'

# # algorithsm to plot
# algorithms=['cart', 'dl85', 'gosdt_warm_lb']

# algorithms_guess=['gosdt_warm_lb']
# algorithms_no_guess = ['dl85', 'gosdt']

# # the depths to plot
# depths = [5, 6, -1]

# # regularization
# regs=[0.001, 0.0002]


# # do the plot
# plot_runtime(figurename, excelpath_guess, excelpath_no_guess, dataset=dataset,
#              algorithms_guess=algorithms_guess,
#              algorithms_no_guess = algorithms_no_guess,
#              depths=depths, regs=regs)
