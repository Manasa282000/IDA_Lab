import sys
from plot import plot_bar

# output file name ('figures/figure7' or 'figures/figure8')
# figure 7 is (thresh=False & lb_guess=True) vs (thresh=False & lb_guess=False)
# figure 8 is (thresh=True & lb_guess=True) vs (thresh=True & lb_guess=False)
figurename='figures/figure7'

# setting the excel path
resultspath = '../results/allresults.csv'

# yaxis metrics
metrics = ['time', 'train_acc', 'test_acc']

# algorithsm to plot
algorithms = ['dl8.5', 'gosdt']

# the depths to plot
depth_budgets = [5,6]

regs = {'spiral':0.01, 'compas':0.001, 'broward_general_2y':0.005, 'coupon_carryout':0.001, 
              'coupon_rest20':0.001, 'fico':0.0005, 'netherlands':0.001}

max_depths = {'spiral':3, 'compas':3, 'broward_general_2y':1, 'coupon_carryout':2, 
              'coupon_rest20':2, 'fico':1, 'netherlands':2}
n_ests = {'spiral':20, 'compas':20, 'broward_general_2y':40, 'coupon_carryout':50, 
              'coupon_rest20':50, 'fico':40, 'netherlands':30}

lb_max_depths = {'spiral':1, 'compas':1, 'broward_general_2y':1, 
        'coupon_carryout':1,'coupon_rest20':1,
        'fico': 1, 'netherlands':1}
lb_n_ests = {'spiral':100,'compas':100, 'broward_general_2y':100, 
        'coupon_carryout':100,'coupon_rest20':100,
        'fico': 100, 'netherlands':100}

for algorithm in algorithms:
    for depth_budget in depth_budgets:
        datasets = ['spiral', 'compas', 'broward_general_2y', 'coupon_carryout', 'coupon_rest20', 'fico', 'netherlands']
        for y_axis in metrics:
            plot_bar(figurename, resultspath, algorithm, depth_budget, 
                      datasets, regs, y_axis, max_depths, n_ests, max_depths, n_ests)


# import sys
# from plot import plot_bar_guessing_lb

# # output file name
# figurename='figures/figure7'

# # setting the excel path
# excelpath = '../results/combined_data.xlsx'

# # yaxis metrics
# metrics = ['Training Time', 'Training Accuracy', 'Test Accuracy']

# # the datasets
# dataset='2y_score'

# # algorithsm to plot
# algorithms=['gosdt', 'dl85']

# # the depths to plot
# depths = [5]

# # Figure 4
# for y_axis in metrics:
#     for depth in depths:
#         plot_bar_guessing_lb(figurename, excelpath, d=depth, y_axis=y_axis)
