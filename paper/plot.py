from numpy.lib.scimath import log
import pandas as pd
import numpy as np
import matplotlib
from matplotlib import pyplot as plt
from matplotlib.collections import PatchCollection
from matplotlib.patches import Rectangle
from matplotlib.container import ErrorbarContainer
from matplotlib.lines import Line2D
from matplotlib.collections import LineCollection
import matplotlib.patches as mpatches
import json

CONFIG_REFERENCE_ACC='../results/reference_acc.json'

# Figure 3
def plot_tradeoff(figurename, resultspath, batreepath, dataset, guess_type,
                  max_depth=None, n_est=None, lb_max_depth=None, lb_n_est=None):
    """
    Parameters
    ----------
    guess_type : string
        thresh or lb_guess or both.

    """
    df = pd.read_csv(resultspath)            
    df = df[df['dataset']==dataset]
    with open(CONFIG_REFERENCE_ACC, 'r') as f:
        ref = json.loads(f.read())
    for y_axis in ['train_acc', 'test_acc']:
        if (dataset == 'spiral') and (y_axis == 'test_acc'):
            continue
        ref_acc = np.median(ref[dataset][y_axis])
        title = ref[dataset]['title']

        # Styling
        plt.rc('font', size = 18)
        colors = [ '#d77ede', '#e3a042', '#509eba']
        markers = ['o', 's', 'D']

        # Axis Labels
        x_axis = 'leaves'

        x_max = 0

        # Resolution + Size
        plt.figure(figsize=(8, 5), dpi=80)

        #dataframe = dataset

        # Go through algorithms in alphabetic order, plot each algorithm's curve
        algorithms = ['cart', 'dl8.5', 'gosdt']
        for i, algorithm in enumerate(sorted(set(algorithms))):
            if algorithm == 'cart':
                data = df[(df['thresh']==False) & (df['lb_guess']==False) & (df['algorithm']=='cart')]
            else:
                if guess_type == 'thresh':
                    data = df[(df['thresh']==True) & (df['lb_guess']==False) & (df['algorithm']==algorithm) & 
                              (df['max_depth']==max_depth) & (df['n_est']==n_est)]
                    l = 'th'
                elif guess_type == 'lb_guess':
                    data = df[(df['thresh']==False) & (df['lb_guess']==True) & (df['algorithm']==algorithm) & 
                              (df['lb_max_depth']==lb_max_depth) & (df['lb_n_est']==lb_n_est)]
                    l = 'lb'
                else:
                    data = df[(df['thresh']==True) & (df['lb_guess']==True) & (df['algorithm']==algorithm) & 
                              (df['max_depth']==max_depth) & (df['n_est']==n_est) &
                              (df['lb_max_depth']==lb_max_depth) & (df['lb_n_est']==lb_n_est)]
                    l = 'th+lb'
                #data = df[(df['thresh']==True) & (df['lb_guess']==True) & (df['algorithm']==algorithm)]
            if data.empty:
                print('data is empty')
                continue
            
            # Iterate through each unique configuration
            depths = sorted(set(data[data['optimaldepth']==0]['depth_budget'].values))
            if algorithm != 'gosdt':
                if 0 in depths:
                    depths.remove(0)
            for depth in depths:
                x = []; y = []
                x_low = []; x_high = []
                y_low = []; y_high = []
                errorboxes = []
                points = set()
                for reg in sorted(set(data['regularization'].values)):
                    point_results = data[(data['depth_budget'] == depth) & (data['regularization'] == reg) & 
                                         ((data['runstatus'] == 'SUCCESS') | (data['runstatus'] == 'ALGORITHM_TIMEOUT'))]
                    
                    # only plot the points from SUCCESS run without timeout
                    if (point_results['time'] == -1).any():
                        continue

                    x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
                    y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])

                    point = (
                        x_iqr[0.5], y_iqr[0.5],
                        x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
                        y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

                    if x_iqr[0.75] > x_max:
                        x_max = x_iqr[0.75]

                    if not point in points:
                        points.add(point)
                        x.append(point[0]); y.append(point[1])
                        x_low.append(point[2]); x_high.append(point[3])
                        y_low.append(point[4]); y_high.append(point[5])

                for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
                    rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
                    errorboxes.append(rect)

                if algorithm == 'gosdt' and depth==0:
                    c = '#233c82'
                else:
                    c = colors[i]

                plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high],
                    markersize=10, marker=markers[i],
                    color=c, alpha=0.75,
                    linewidth=1, linestyle='none')



        df_ba = pd.read_csv(batreepath)
        data = df_ba[df_ba['dataset'] == dataset]
        x = []; y = []
        x_low = []; x_high = []
        y_low = []; y_high = []
        errorboxes = []
        points = set()

        point_results = data[(data['d'] == 3)&(data['n_trees'] == 10)&(data['objective']==4)]
        x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
        y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])
        point = (x_iqr[0.5], y_iqr[0.5],
                 x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
                 y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

        if x_iqr[0.75] > x_max:
            x_max = x_iqr[0.75]

        if not point in points:
            points.add(point)
            x.append(point[0]); y.append(point[1])
            x_low.append(point[2]); x_high.append(point[3])
            y_low.append(point[4]); y_high.append(point[5])


        for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
            rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
            errorboxes.append(rect)

        # c = '#c44e52'
        # l = 'batree'
        # m = '^'
        plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high], label='batree',
                    markersize=10, marker='^',
                    color='#c44e52', alpha=1,
                    linewidth=1, linestyle='none')
        plt.hlines(ref_acc, 0, x_max, color='black', ls='--')
        plt.xscale('log')
        plt.xlabel('Number of Leaves')
        if y_axis == 'train_acc':
            ylabel = 'Training Accuracy'
        else:
            ylabel = 'Test Accuracy'
        plt.ylabel(ylabel+" (%)")

        handles = []
        labels = ['cart', 'dl8.5+'+l, 'gosdt+'+l]
        if 0 in depths:
            colors.append('#233c82')
            markers.append('D')
            labels.append('gosdt (no depth)+'+l)
        colors.append('#c44e52')
        markers.append('^')
        labels.append('batree')
        for j in range(len(colors)):
            line = Line2D([], [], color=colors[j], marker=markers[j], markersize=8, ls='none')
            barline = LineCollection(np.empty((1,2,2)), colors=c, linewidths=1)
            handles.append(ErrorbarContainer((line, (), [barline]), has_xerr=True, has_yerr=True))
        handles.append(Line2D([], [], color='black', ls='--'))
        labels.append('GBDT')
        plt.legend(handles, labels, prop={'size': 14})

        plt.title("{} vs Number of Leaves\n({})".format(ylabel, title))
        plt.savefig("{}_{}_vs_leaves_{}_{}_max_depth{}_n_est{}_lb_max_depth{}_lb_n_est{}.png".format(figurename, y_axis.lower().replace(' ', '_'), title, guess_type, max_depth, n_est, lb_max_depth, lb_n_est), bbox_inches='tight')
        plt.savefig("{}_{}_vs_leaves_{}_{}_max_depth{}_n_est{}_lb_max_depth{}_lb_n_est{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), title, guess_type, max_depth, n_est, lb_max_depth, lb_n_est), bbox_inches='tight')


# Figure 2
def plot_runtime(figurename, resultspath, dataset, depth_budgets, regs, 
                 max_depth=None, n_est=None, lb_max_depth=None, lb_n_est=None):
    df = pd.read_csv(resultspath)
    df = df[df['dataset']==dataset]
    df_no_guess = df[(df['thresh']==False) & (df['lb_guess']==False) & (df['optimaldepth']==0)]
    df_guess = df[(df['thresh']==True) & (df['lb_guess']==True) & (df['optimaldepth']==0) & 
                  (df['max_depth']==max_depth) & (df['n_est']==n_est) &
                  (df['lb_max_depth']==lb_max_depth) & (df['lb_n_est']==lb_n_est)]
    algorithms = ['dl8.5', 'gosdt']
    
    with open(CONFIG_REFERENCE_ACC, 'r') as f:
        ref = json.loads(f.read())
    for y_axis in ['train_acc', 'test_acc']:
        plt.rc('font', size = 18)

        ref_acc = np.median(ref[dataset][y_axis])
        title = ref[dataset]['title']

        x_axis = "time"

        fig, axs = plt.subplots(1,len(depth_budgets), figsize=(9*len(depth_budgets), 5))
        axs = axs.ravel()
        
        colors = ['#e3a042', '#509eba']
        markers = ['.', '*']
        
        
        for j, depth in enumerate(depth_budgets):
            for k, dataframe in enumerate([df_guess, df_no_guess]):
                for i, algorithm in enumerate(sorted(set(algorithms))):
                    if depth == 0 and algorithm == 'dl8.5':
                        continue
                    data = dataframe[dataframe['algorithm'] == algorithm]
                    x = []; y = []
                    x_low = []; x_high = []
                    y_low = []; y_high = []
                    errorboxes = []
                    points = set()
                    sparsity = []
                    na_index = []
                    timeout_index = []

                    for reg in regs:
                        point_results = data[(data['depth_budget']==depth) & (data['regularization']==reg)]
                        if (point_results['runstatus'] == 'FAIL_SLURM_TIMEOUT').any() or (point_results['runstatus'] == 'FAIL_SLURM_OUT_OF_MEMORY').any():
                            sparsity.append(10)
                            point_results[x_axis]=1800
                            point_results[y_axis] = 0.5
                            na_index.append(len(x))
                        else:
                        #elif (point_results['runstatus'] == 'SUCCESS').all() or (point_results['runstatus'] == 'ALGORITHM_TIMEOUT').all():
                            #point_results = data[(data['depth_budget']==depth) & (data['regularization']==reg) & (data['runstatus']=='SUCCESS')]
                            sparsity.append(np.median(point_results['leaves']))
                            if (point_results[x_axis]==-1).any():
                                point_results.loc[point_results[x_axis] == -1, x_axis] = 1800
                                timeout_index.append(len(x))
                        #else:
                        #    print('check runstatus')
                        #    return
                        x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
                        y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])

                        point = (
                            x_iqr[0.5], y_iqr[0.5],
                            x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
                            y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

                        #if not point in points:
                        points.add(point)
                        x.append(point[0]); y.append(point[1])
                        x_low.append(point[2]); x_high.append(point[3])
                        y_low.append(point[4]); y_high.append(point[5])

                    for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
                        rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
                        errorboxes.append(rect)
                    #print('na_index:', na_index)
                    #print('timeout_index:', timeout_index)
                    for index in range(len(x)):
                        if index in na_index:
                            axs[j].errorbar(x[index], y[index], xerr=np.array([x_low[index], x_high[index]]).reshape(2,1), 
                                        yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
                                     markersize=sparsity[index]+10, marker=markers[k],
                                     color=colors[i], alpha=0.3,
                                     linewidth=1, linestyle='dashed')
                        elif index in timeout_index:
                            axs[j].errorbar(x[index], y[index], xerr=np.array([x_low[index], x_high[index]]).reshape(2,1), 
                                        yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
                                     markersize=sparsity[index]+10, marker=markers[k],
                                     color=colors[i], alpha=1,
                                     linewidth=1, linestyle='none')
                        else:
                            axs[j].errorbar(x[index], y[index], xerr=np.array([x_low[index], x_high[index]]).reshape(2,1), 
                                        yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
                                     markersize=sparsity[index]+10, marker=markers[k],
                                     color=colors[i], alpha=1,
                                     linewidth=1, linestyle='none')


            axs[j].set_xscale('log')
            axs[j].set_xlabel('Training Time (s)')
            if y_axis == 'train_acc':
                ylabel = 'Training Accuracy'
            else:
                ylabel = 'Test Accuracy'
            axs[j].set_ylabel(ylabel + ' (%)')
                
            if depth == 0:
                axs[j].set_title("{} vs Run Time\n{} (no depth constraint)".format(ylabel, title))
            else:
                axs[j].set_title("{} vs Run Time\n{} (depth limit={})".format(ylabel, title, depth-1))
            
            axs[j].axhline(ref_acc, color='black', ls='--')
           
        handles = []
        for c in ['#e3a042', '#509eba']:
            for m in ['.','*']:
                line = Line2D([], [], color=c, marker=m, markersize=18, ls='none')
                barline = LineCollection(np.empty((1,2,2)), colors=c, linewidths=1)
                handles.append(ErrorbarContainer((line, (), [barline]), has_xerr=True, has_yerr=True))
        handles.append(Line2D([], [], color='black', ls='--'))
        labels = ['dl8.5+th+lb', 'dl8.5', 'gosdt+th+lb', 'gosdt', 'GBDT']
        axs[j].legend(handles, labels, prop={'size': 14})#,bbox_to_anchor=(1, 1)  )

        plt.savefig("{}_{}_vs_runtime_{}_max_depth{}_n_est{}_lb_max_depth{}_lb_n_est{}.png".format(figurename, y_axis.lower().replace(' ', '_'), title, max_depth, n_est, lb_max_depth, lb_n_est), bbox_inches='tight')
        plt.savefig("{}_{}_vs_runtime_{}_max_depth{}_n_est{}_lb_max_depth{}_lb_n_est{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), title, max_depth, n_est, lb_max_depth, lb_n_est), bbox_inches='tight')



# Figure 4 & Figure 7 & Figure 8
# Figure 8 is gosdt+th+lb vs gosdt+th
def plot_bar(figurename, resultspath, algorithm, depth_budget, 
             datasets, regs, y_axis, max_depths=None, n_ests=None, 
             lb_max_depths=None, lb_n_ests=None):
    '''
    Parameters
    ----------
    figurename : string
    resultspath : string
        path to allresults.csv.
    algorithm : string
        dl8.5 or gosdt.
    depth_budget : int
    datasets : list
        name of all datasets.
    max_depths : dictionary
        max_depth of each dataset (either threshold guess or lower bound guess).
    n_ests : dictionary
        n_est of each dataset (either threshold guess or lower bound guess).
    regs : dictionary
        regularization of each dataset.
    y_axis : string
        train_acc, test_acc, time.
    '''
    
    points_guess = []
    points_no_guess = []
    guess_time = []
    no_guess_timeout_index = []
    guess_timeout_index = []
    
    df = pd.read_csv(resultspath)
    with open(CONFIG_REFERENCE_ACC, 'r') as f:
        ref = json.loads(f.read())
    title = [ref[dataset]['title'] for dataset in datasets]
    title = [title[i] + '\nλ='+str(regs[dataset]) for i, dataset in enumerate(datasets)]
    
    if y_axis == 'test_acc':
        plt.figure(figsize=(7.5,4))
        if 'spiral' in datasets:
            idx = datasets.index('spiral')
            datasets.remove('spiral')
            del title[idx]
    else:
        plt.figure(figsize=(8.5,4))
    
    index = np.arange(len(datasets))
    width=0.4
    for k, dataset in enumerate(datasets):
        reg = regs[dataset]
        
        
        if 'figure4' in figurename:
            max_depth = max_depths[dataset]
            n_est = n_ests[dataset]
            guess = df[(df['dataset']==dataset) & (df['thresh']==True) & (df['lb_guess']==False) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) &
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg) & 
                        (df['max_depth']==max_depth) & (df['n_est']==n_est)]

            no_guess = df[(df['dataset']==dataset) & (df['thresh']==False) & (df['lb_guess']==False) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) & 
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg)]
            l = 'th'
            g_time = 'guesstime'
        elif 'figure7' in figurename:
            lb_max_depth = lb_max_depths[dataset]
            lb_n_est = lb_n_ests[dataset]
            guess = df[(df['dataset']==dataset) & (df['thresh']==False) & (df['lb_guess']==True) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) & 
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg) & 
                        (df['lb_max_depth']==lb_max_depth) & (df['lb_n_est']==lb_n_est)]

            no_guess = df[(df['dataset']==dataset) & (df['thresh']==False) & (df['lb_guess']==False) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) & 
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg)]
            l = 'lb'
            g_time = 'lb_time'
        elif 'figure8' in figurename:
            max_depth = max_depths[dataset]
            n_est = n_ests[dataset]
            lb_max_depth = lb_max_depths[dataset]
            lb_n_est = lb_n_ests[dataset]
            guess = df[(df['dataset']==dataset) & (df['thresh']==True) & (df['lb_guess']==True) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) & 
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg) & 
                        (df['max_depth']==max_depth) & (df['n_est']==n_est) &
                        (df['lb_max_depth']==lb_max_depth) & (df['lb_n_est']==lb_n_est)]

            no_guess = df[(df['dataset']==dataset) & (df['thresh']==True) & (df['lb_guess']==False) &
                        (df['algorithm']==algorithm) & (df['optimaldepth']==0) & 
                        (df['depth_budget']==depth_budget) & (df['regularization']==reg) &
                        (df['max_depth']==max_depth) & (df['n_est']==n_est)]
            l = 'lb'
            g_time = 'lb_time'
        else:
            print('Check figurename and plot function!')
            return
        
        if y_axis=='time':
            if (no_guess['runstatus'] == 'FAIL_SLURM_OUT_OF_MEMORY').any() or (no_guess['runstatus'] == 'FAIL_SLURM_TIMEOUT').any():
                points_no_guess.append(-1)
            else: #(no_guess['runstatus'] == 'SUCCESS').all():
                no_guess.loc[((no_guess[y_axis]==-1) | (no_guess[y_axis] > 1800)), y_axis] = 1800
                points_no_guess.append(np.mean(no_guess[y_axis]))
                
            if (guess['runstatus'] == 'FAIL_SLURM_OUT_OF_MEMORY').any() or (guess['runstatus'] == 'FAIL_SLURM_TIMEOUT').any():
                points_guess.append(-1)
                guess_time.append(-1)
            else: #(guess['runstatus'] == 'SUCCESS').all():
                guess.loc[((guess[y_axis]==-1) | (guess[y_axis] > 1800)), y_axis] = 1800
                points_guess.append(np.mean(guess[y_axis]))
                guess_time.append(np.mean(guess[g_time]))

        else:
            
            if (no_guess['runstatus'] == 'FAIL_SLURM_OUT_OF_MEMORY').any() or (no_guess['runstatus'] == 'FAIL_SLURM_TIMEOUT').any():
                points_no_guess.append(0.5)
            else: #(no_guess['runstatus'] == 'SUCCESS').all():
                if no_guess.loc[((no_guess['time']==-1) | (no_guess['time'] > 1800)),].shape[0] > 0:
                    no_guess_timeout_index.append(len(points_no_guess))
                points_no_guess.append(np.mean(no_guess[y_axis]))
            
            if (guess['runstatus'] == 'FAIL_SLURM_OUT_OF_MEMORY').any() or (guess['runstatus'] == 'FAIL_SLURM_TIMEOUT').any():
                points_guess.append(0.5)
            else: #(guess['runstatus'] == 'SUCCESS').all():
                if guess.loc[((guess['time']==-1) | (guess['time'] > 1800)),].shape[0] > 0:
                    guess_timeout_index.append(len(points_guess))
                points_guess.append(np.mean(guess[y_axis]))
            

    points_guess = np.array(points_guess)
    points_no_guess = np.array(points_no_guess)
    guess_time = np.array(guess_time)
    no_guess_timeout_index = np.array(no_guess_timeout_index)
    guess_timeout_index = np.array(guess_timeout_index)
    
    if y_axis == 'time':
        colors = ['#4c72b0', '#dd8452']
        cs = []
        labels = []
        hatchs = []
        for u, plot_points in enumerate([points_guess, points_no_guess]):
            
            i = np.where(plot_points == -1)[0] # memory out or slurm time out
            j = np.where(plot_points == 1800)[0] # algorithm time out
            k = np.where((plot_points != -1) & (plot_points < 1800))[0]
            
            if u == 0:
                if 'figure8' in figurename:
                    tentative_label = algorithm + '+th+' +l
                else:
                    tentative_label = algorithm + '+' + l
            else:
                if 'figure8' in figurename:
                    tentative_label = algorithm + '+th'
                else:
                    tentative_label = algorithm
            
            if len(k) != 0:
                if u==0:
                    plt.bar(index[k] - width/2, plot_points[k], width, bottom=0, color=colors[u]) # training time + guessing 
                    plt.bar(index[k] - width/2, guess_time[k], width, bottom=plot_points[k], color='#2ca02c') # training time + guessing 
                    if 'guess_time' not in labels:
                        cs.append('#2ca02c')
                        labels.append('guess time')
                        hatchs.append(None)
                else:
                    plt.bar(index[k] + width/2, plot_points[k], width, bottom=0, color=colors[u]) # training time + guessing                     
                if tentative_label not in labels:
                    cs.append(colors[u])
                    labels.append(tentative_label)
                    hatchs.append(None)
            if len(j) > 0:
                if u==0:
                    plt.bar(index[j] - width/2, plot_points[j], width, bottom=0, color=colors[u], hatch='/') # training time + guessing algorithm timeout
                    plt.bar(index[j] - width/2, guess_time[j], width, bottom=plot_points[j], color='#2ca02c', hatch='/') # training time + guessing algorithm timeout
                else:
                    plt.bar(index[j] + width/2, plot_points[j], width, bottom=0, color=colors[u], hatch='/') # training time + guessing algorithm timeout
                    
                if tentative_label + ' time out' not in labels:
                    cs.append(colors[u])
                    labels.append(tentative_label+ ' time out')
                    hatchs.append(r'///')
            if len(i) > 0:
                if u==0:
                    plt.bar(index[i] - width/2, np.repeat(1800, len(i)), width, bottom=0, color='silver', hatch='/') # NA memory out or slurm time out
                    plt.bar(index[i] - width/2, np.repeat(30, len(i)), width, bottom=plot_points[i], color='silver', hatch='/') # NA memory out or slurm time out
                else:
                    plt.bar(index[i] + width/2, np.repeat(1830, len(i)), width, bottom=0, color='silver', hatch='/') # NA memory out or slurm time out
                if 'N/A' not in labels:
                    cs.append('silver')
                    labels.append('N/A')
                    hatchs.append(r'///')
            
        ylabel = 'Training Time (s)'
        plt.ylabel(ylabel, fontsize=20)
        plt.yscale('log')
        plt.ylim(1, 2000)

        handles = [mpatches.Patch(facecolor=cs[i], hatch = hatchs[i]) for i in range(len(cs))]
        plt.legend(handles, labels, prop={'size': 14}, bbox_to_anchor=(1.01, 1))

    else:
        colors = ['#ccb974', '#8172b3']
        cs = []
        labels = []
        hatchs = []
        for u, plot_points in enumerate([points_guess, points_no_guess]):
            
            i = np.where(plot_points == 0.5)[0] # memory out or slurm time out
            j = np.arange(len(plot_points))
                   
            if u == 0:
                if 'figure8' in figurename:
                    tentative_label = algorithm + '+th+' + l
                else:
                    tentative_label = algorithm + '+' + l
                timeout_index = guess_timeout_index 
            else:
                if 'figure8' in figurename:
                    tentative_label = algorithm+'+th'
                else:
                    tentative_label = algorithm
                timeout_index = no_guess_timeout_index
                
            if len(np.append(i, timeout_index))!=0:
                j = np.delete(j, np.append(i, timeout_index).astype('int64')) # index effective accuracy
                    
            if len(j) != 0:
                if tentative_label not in labels:
                    cs.append(colors[u])
                    labels.append(tentative_label)
                    hatchs.append(None)
                if u == 0:
                    plt.bar(index[j] - width/2, plot_points[j], width, bottom=0, color=colors[u]) # accuracy + guessing 
                else:
                    plt.bar(index[j] + width/2, plot_points[j], width, bottom=0, color=colors[u]) # accuracy + guessing 

            if len(timeout_index) > 0:
                if tentative_label +' time out' not in labels:
                    cs.append(colors[u])
                    labels.append(tentative_label + ' time out')
                    hatchs.append(r'///')
                if u == 0:
                    plt.bar(index[guess_timeout_index] - width/2, plot_points[guess_timeout_index], width, bottom=0, color=colors[u], hatch='/') # accuracy + guessing algorithm timeout
                else:
                    plt.bar(index[no_guess_timeout_index] + width/2, plot_points[no_guess_timeout_index], width, bottom=0, color=colors[u], hatch='/') # accuracy + guessing algorithm timeout
            if len(i) > 0:
                if 'N/A' not in labels:
                    cs.append('silver')
                    labels.append('N/A')
                    hatchs.append(r'///')
                if u == 0:
                    plt.bar(index[i] - width/2, plot_points[i], width, bottom=0, color='silver', hatch='/') # NA memory out or slurm time out
                else:
                    plt.bar(index[i] + width/2, plot_points[i], width, bottom=0, color='silver', hatch='/') # NA memory out or slurm time out
                    
          
        if y_axis == 'train_acc':
            ylabel = 'Training Accuracy'
        else:
            ylabel = 'Test Accuracy'
        plt.ylabel(ylabel, fontsize=20)

        handles = [mpatches.Patch(facecolor=cs[i], hatch = hatchs[i]) for i in range(len(cs))]
        plt.legend(handles, labels, prop={'size': 14}, bbox_to_anchor=(1.01,1))

    plt.yticks(fontsize=15)
    plt.xticks(index, title, fontsize=18, rotation=-90)
    if depth_budget != 0:
        plt.title('{} {} \ndepth limit={}'.format(algorithm, ylabel.lower(), depth_budget-1), fontsize=20)
    else:
        plt.title('{} {} \nno depth constraint'.format(algorithm, ylabel.lower()), fontsize=20)
    if 'figure4' in figurename:
        plt.savefig("{}_threshold_{}_{}_depth_budget_{}_max_depth{}_n_est{}.png".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), max_depths['compas'], n_ests['compas']), bbox_inches='tight')
        plt.savefig("{}_threshold_{}_{}_depth_budget_{}_max_depth{}_n_est{}.pdf".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), max_depths['compas'], n_ests['compas']), bbox_inches='tight')
    elif 'figure7' in figurename:
        plt.savefig("{}_lb_{}_{}_depth_budget_{}_lb_max_depth{}_lb_n_est{}.png".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), lb_max_depths['compas'], lb_n_ests['compas']), bbox_inches='tight')
        plt.savefig("{}_lb_{}_{}_depth_budget_{}_lb_max_depth{}_lb_n_est{}.pdf".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), lb_max_depths['compas'], lb_n_ests['compas']), bbox_inches='tight')
    elif 'figure8' in figurename:
        plt.savefig("{}_th_lb_{}_{}_depth_budget_{}_lb_max_depth{}_lb_n_est{}.png".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), lb_max_depths['compas'], lb_n_ests['compas']), bbox_inches='tight')
        plt.savefig("{}_th_lb_{}_{}_depth_budget_{}_lb_max_depth{}_lb_n_est{}.pdf".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(depth_budget), lb_max_depths['compas'], lb_n_ests['compas']), bbox_inches='tight')
    else:
        print('check figurename and plot function!')

# Figure 6
def plot_depth_guess(figurename, resultspath, dataset):
    algorithm = 'gosdt'
    df = pd.read_csv(resultspath)
    df = df[(df['dataset']==dataset) & (df['algorithm']==algorithm) & (df['optimaldepth'] != 0)]
    with open(CONFIG_REFERENCE_ACC, 'r') as f:
        ref = json.loads(f.read())
    for y_axis in ['time', 'Accuracy']:
        
        
        for i, reg in enumerate(sorted(set(df['regularization'].values))):
            df_new = df[(df['regularization']==reg)]
            for j, opt_depth in enumerate(sorted(set(df_new['optimaldepth'].values))):
                plt.rc('font', size = 18)

                title = ref[dataset]['title']
        
                x_axis = "depth_budget"
                plt.figure(figsize=(9,5))
        
                if y_axis == 'Accuracy':
                    z=[]; z_low = []; z_high = [] # store test accuracy
                y=[]; y_low = []; y_high = []
                
                data = df_new[df_new['optimaldepth']==opt_depth]
                depth_est = data['depth_budget'].values
                eff_depth_est = list(depth_est)
                xticks = depth_est-opt_depth

                for depth in depth_est:
                    point_results = data[(data['depth_budget'] == depth) & 
                                         ((data['runstatus'] == 'SUCCESS') | (data['runstatus'] == 'ALGORITHM_TIMEOUT')) & 
                                         (data['thresh']==True) & (data['lb_guess']==False)]
                    if point_results.empty:
                        eff_depth_est.remove(depth)
                        continue
                    if y_axis == 'Accuracy':
                        y_iqr = point_results['train_acc'].quantile([0.25, 0.5, 0.75])
                        z_iqr = point_results['test_acc'].quantile([0.25, 0.5, 0.75])
                        z.append(z_iqr[0.5]); z_low.append(z_iqr[0.5]-z_iqr[0.25]);
                        z_high.append(z_iqr[0.75]-z_iqr[0.5])
                    else:
                        point_results.loc[point_results[y_axis]==-1, y_axis]=1800
                        y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])
                    point = (y_iqr[0.5],
                        y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])
        
                    y.append(point[0])
                    y_low.append(point[1]); y_high.append(point[2])
        
                depth_off = np.array(eff_depth_est) - opt_depth
                for index, depth in enumerate(eff_depth_est):
                    if depth == opt_depth:
                        c = 'C1'
                    else:
                        c = 'C0'
                    plt.errorbar(depth_off[index], y[index],
                                 yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
                                 markersize=15, marker='o',
                                 color=c, alpha=1,
                                 linewidth=1, linestyle='none')
                    if y_axis == 'Accuracy':
                        plt.errorbar(depth_off[index], z[index],
                                 yerr=np.array([z_low[index], z_high[index]]).reshape(2,1),
                                 markersize=15, marker='^',
                                 color=c, alpha=1,
                                 linewidth=1, linestyle='none')
                plt.grid(color='0.93')
                plt.xticks(xticks)
                plt.xlabel('Offset from optimal depth={}'.format(opt_depth-1))
                plt.ylabel(y_axis)
                plt.title("{} vs Depth Limit\n{}".format(y_axis, title))
        
                handles = []
                for c in ['C1', 'C0']:
                    line = Line2D([], [], color=c, marker='o', markersize=10, ls='none')
                    handles.append(line)
                    if y_axis == 'Accuracy':
                        line = Line2D([], [], color=c, marker='^', markersize=10, ls='none')
                        handles.append(line)
                if y_axis == 'Accuracy':
                    labels = ['training accuracy at optimal depth', 'test accuracy at optimal depth',
                              'training accuracy at suboptimal depth', 'test accuracy at suboptimal depth']
                else:
                    labels = ['optimal depth', 'suboptimal depth']
                plt.legend(handles, labels, prop={'size': 14})#,bbox_to_anchor=(1, 1)  )
        
                plt.savefig("{}_depth_guess_{}_vs_{}_{}_{}_{}.png".format(figurename, y_axis.lower().replace(' ', '_'), x_axis.lower().replace(' ','_'), title, opt_depth, reg), bbox_inches='tight')
                plt.savefig("{}_depth_guess_{}_vs_{}_{}_{}_{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), x_axis.lower().replace(' ','_'), title, opt_depth, reg), bbox_inches='tight')
        
# Figure 5
def plot_depth(figurename, resultspath, dataset, reg = 0.001, max_depth=None, n_est=None):
    #leftover parameters from when we potted multiple algorithms, guesses
    algorithm = 'gosdt'
    guess_type = 'thresh'
    i = 0

    #make sure we don't use type 3 fonts: 
    #matplotlib.rcParams['pdf.fonttype'] = 42
    matplotlib.rcParams['text.usetex'] = False

    df = pd.read_csv(resultspath)            
    df = df[df['dataset']==dataset]
    for y_axis in ['train_acc', 'test_acc', 'time']:
        if (dataset == 'spiral') and (y_axis == 'test_acc'):
            continue
        #title = str(dataset) + ', lambda=' + str(reg)
        #subtitle = 'with threshold guessing, n_est='+ str(n_est) + ', max_depth=' + str(max_depth)

        # Styling
        plt.rc('font', size = 18)
        colors = ['C0']
        markers = ['o']

        # Resolution + Size
        if (y_axis == 'time'): 
            plt.figure(figsize=(4.5,5), dpi=150)
        else: 
            plt.figure(figsize=(4,5), dpi=150)

        #dataframe = dataset

        data = df[(df['thresh']==True) & (df['lb_guess']==False) & (df['algorithm']==algorithm) & 
                      (df['max_depth']==max_depth) & (df['n_est']==n_est) & (df['regularization'] == reg)]
        if data.empty:
            print('data is missing')
            return
            
        # Iterate through each unique configuration
        depths = sorted(set(data['depth_budget'].values))
        x_max = max(depths)
        for depth in depths:
            x = []
            y = []
            y_low = []; y_high = []
            errorboxes = []
            points = set()
            #for reg in sorted(set(data['regularization'].values)):
            point_results = data[(data['depth_budget'] == depth) & (data['regularization'] == reg) & 
                                    ((data['runstatus'] == 'SUCCESS') | (data['runstatus'] == 'ALGORITHM_TIMEOUT'))]
            
            # only plot the points from SUCCESS run without timeout
            if (point_results['time'] == -1).any():
                continue

            y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])

            point = (
                depth, y_iqr[0.5],
                y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

            if not point in points:
                points.add(point)
                x.append(point[0]); y.append(point[1])
                y_low.append(point[2]); y_high.append(point[3])

            for x, yv, yl, yh in zip(x, y, y_low, y_high):
                rect = Rectangle((0, yv - yl), 0, max(yl + yh, 0.00001))
                errorboxes.append(rect)

            if algorithm=='gosdt' and depth==0:
                c = 'C1'
                plt.hlines(y, 0, x_max, color=c, ls='--')
            else:
                c = colors[i]

            plt.errorbar(x, y, yerr=[y_low, y_high],
                markersize=10, marker=markers[i],
                color=c, alpha=0.75,
                linewidth=1, linestyle='none')

        plt.xlabel('Depth Limit', fontsize=20)#(' + "$\infty$" + '=no limit)', fontsize=24

        #set up to plot depth limit 0 on the right, with label infty
        #xticks = copy.deepcopy(depths)
        #xticks.remove(0)
        #xticks = xticks + [x_max + 1]
        #plt.xticks(xticks, ['2', '3','4','5','6','7','8', r"$\infty$"])

        #set up to plot depth limit 0 on the left, with label infty
        xticks = depths
        plt.xticks(xticks, [r"$\infty$", '2', '3','4','5','6','7','8'], fontsize=24)
        if y_axis == 'train_acc':
            ylabel = 'Train Acc'
            plt.ylim(0.64, 0.695)
            plt.yticks([0.64, 0.65, 0.66, 0.67, 0.68, 0.69], fontsize=20)
        elif y_axis == 'test_acc':
            ylabel = 'Test Acc'
            plt.ylim(0.64, 0.695)
            plt.yticks([0.64, 0.65, 0.66, 0.67, 0.68, 0.69], fontsize=20)
        else:
            ylabel = 'Train Time(s)' 
            # plt.yscale('log')
            # plt.ylim(0.0000000000000001, 1000000000)

        handles = []
        labels = ['gosdt']
        if 0 in depths:
            colors.append('C1')
            markers.append('o')
            labels.append('gosdt (no depth)')
        for j in range(len(colors)):
            line = Line2D([], [], color=colors[j], marker=markers[j], markersize=8, ls='none')
            barline = LineCollection(np.empty((1,2,2)), colors=colors[j], linewidths=1)
            handles.append(ErrorbarContainer((line, (), [barline]), has_yerr=True))
        if y_axis == 'test_acc' or y_axis == 'train_acc': 
            print("skipping legend for accuracies")
            #plt.legend(handles, labels, prop={'size': 18}, loc = 'lower right')
        else:
            plt.legend(handles, labels, prop={'size': 18}, loc = 'upper center')

        plt.title("{} vs Depth Limit".format(ylabel), fontsize=20)
        plt.savefig("{}_{}_vs_depth_{}_max_depth{}_n_est{}.png".format(figurename, y_axis.lower().replace(' ', '_'), guess_type, max_depth, n_est), bbox_inches='tight')
        plt.savefig("{}_{}_vs_depth_{}_max_depth{}_n_est{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), guess_type, max_depth, n_est), bbox_inches='tight')
               


# import pandas as pd
# import numpy as np
# from matplotlib import pyplot as plt
# from matplotlib.collections import PatchCollection
# from matplotlib.patches import Rectangle
# from matplotlib.container import ErrorbarContainer
# from matplotlib.lines import Line2D
# from matplotlib.collections import LineCollection
# import matplotlib.patches as mpatches
# import json

# CONFIG_REFERENCE_ACC='../results/reference_acc.json'

# # Figure 3
# def plot_tradeoff(figurename, excelpath, excelpath2, batreepath, dataset, algorithms, depths):
#     dataframe = pd.read_excel(excelpath, dataset+'_thresh')
#     dataframe_cart = pd.read_excel(excelpath2, dataset+'_no_thresh')
#     with open(CONFIG_REFERENCE_ACC, 'r') as f:
#         ref = json.loads(f.read())
#     for y_axis in ['Training Accuracy', 'Test Accuracy']:
#         if (dataset == 'spiral') and (y_axis == 'Test Accuracy'):
#             continue
#         ref_acc = np.median(ref[dataset][y_axis])
#         title = ref[dataset]['title']

#         # Styling
#         plt.rc('font', size = 18)
#         colors = [ '#d77ede', '#e3a042', '#509eba']
#         markers = ['o', 's', 'D']

#         # Axis Labels
#         x_axis = '# Leaves'

#         x_max = 0

#         # Resolution + Size
#         plt.figure(figsize=(8, 5), dpi=80)

#         #dataframe = dataset

#         # Go through algorithms in alphabetic order, plot each algorithm's curve
#         for i, algorithm in enumerate(sorted(set(algorithms))):
#             if algorithm == 'cart':
#                 data = dataframe_cart[dataframe_cart['Algorithm'] == algorithm]
#             else:
#                 data = dataframe[dataframe['Algorithm'] == algorithm]
#             # Iterate through each unique configuration
#             for depth in depths:
#                 x = []; y = []
#                 x_low = []; x_high = []
#                 y_low = []; y_high = []
#                 errorboxes = []
#                 points = set()
#                 for reg in sorted(set(data['Regularization'].values)):
#                 #for reg in regs:
#                     point_results = data[(data['Depth Limit'] == depth) & (data['Regularization'] == reg)]
#                     #if len(point_results) <= 2:
#                     #    continue


#                     x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
#                     y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])


#                     point = (
#                         x_iqr[0.5], y_iqr[0.5],
#                         x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
#                         y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

#                     if x_iqr[0.75] > x_max:
#                         x_max = x_iqr[0.75]

#                     if not point in points:
#                         points.add(point)
#                         x.append(point[0]); y.append(point[1])
#                         x_low.append(point[2]); x_high.append(point[3])
#                         y_low.append(point[4]); y_high.append(point[5])

#                 for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
#                     rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
#                     errorboxes.append(rect)

#                 if (algorithm == 'gosdt' or algorithm == 'gosdt_warm_lb') and depth==-1:
#                     c = '#233c82'
#                 else:
#                     c = colors[i]

#                 plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high],
#                     markersize=10, marker=markers[i],
#                     color=c, alpha=0.75,
#                     linewidth=1, linestyle='none')



#         df = pd.read_excel(batreepath, 'alldatasets_cleaned')
#         data = df[df['dataset'] == dataset]
#         x = []; y = []
#         x_low = []; x_high = []
#         y_low = []; y_high = []
#         errorboxes = []
#         points = set()

#         point_results = data[(data['d'] == 3)&(data['n_trees'] == 10)&(data['objective']==4)]
#         x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
#         y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])
#         point = (x_iqr[0.5], y_iqr[0.5],
#                  x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
#                  y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

#         if x_iqr[0.75] > x_max:
#             x_max = x_iqr[0.75]

#         if not point in points:
#             points.add(point)
#             x.append(point[0]); y.append(point[1])
#             x_low.append(point[2]); x_high.append(point[3])
#             y_low.append(point[4]); y_high.append(point[5])


#         for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
#             rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
#             errorboxes.append(rect)

#         c = '#c44e52'
#         l = 'batree'
#         m = '^'
#         plt.errorbar(x, y, xerr=[x_low, x_high], yerr=[y_low, y_high], label=l,
#                     markersize=10, marker=m,
#                     color=c, alpha=1,
#                     linewidth=1, linestyle='none')
#         plt.hlines(ref_acc, 0, x_max, color='black', ls='--')
#         plt.xscale('log')
#         plt.xlabel('Number of Leaves')
#         plt.ylabel(y_axis + " (%)")

#         handles = []
#         labels = ['cart', 'dl8.5+guessing', 'gosdt+guessing']
#         if -1 in depths:
#             colors.append('#233c82')
#             markers.append('D')
#             labels.append('gosdt (no depth)+guessing')
#         colors.append('#c44e52')
#         markers.append('^')
#         labels.append('batree')
#         for j in range(len(colors)):
#             line = Line2D([], [], color=colors[j], marker=markers[j], markersize=8, ls='none')
#             barline = LineCollection(np.empty((1,2,2)), colors=c, linewidths=1)
#             handles.append(ErrorbarContainer((line, (), [barline]), has_xerr=True, has_yerr=True))
#         handles.append(Line2D([], [], color='black', ls='--'))
#         labels.append('GBDT')
#         plt.legend(handles, labels, prop={'size': 14})

#         plt.title("{} vs Number of Leaves\n({})".format(y_axis, title))
#         plt.savefig("{}_{}_vs_leaves_{}.png".format(figurename, y_axis.lower().replace(' ', '_'), title), bbox_inches='tight')
#         plt.savefig("{}_{}_vs_leaves_{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), title), bbox_inches='tight')


# # Figure 2
# def plot_runtime(figurename, excelpath_guess, excelpath_no_guess, dataset, algorithms_guess, algorithms_no_guess, depths, regs):
#     df = [pd.read_excel(excelpath_guess, dataset+'_thresh'),
#             pd.read_excel(excelpath_no_guess, dataset+'_no_thresh')]
#     with open(CONFIG_REFERENCE_ACC, 'r') as f:
#         ref = json.loads(f.read())
#     for y_axis in ['Training Accuracy', 'Test Accuracy']:
#         plt.rc('font', size = 18)

#         ref_acc = np.median(ref[dataset][y_axis])
#         title = ref[dataset]['title']

#         x_axis = "Training Time"

#         fig, axs = plt.subplots(1,len(depths), figsize=(9*len(depths), 5))
#         axs = axs.ravel()

#         for j, depth in enumerate(depths):
#             for k, dataframe in enumerate(df):
#                 if k == 0:
#                     algorithms = algorithms_guess
#                 else:
#                     algorithms = algorithms_no_guess
#                 for i, algorithm in enumerate(sorted(set(algorithms))):
#                     data = dataframe[dataframe['Algorithm'] == algorithm]
#                     x = []; y = []
#                     x_low = []; x_high = []
#                     y_low = []; y_high = []
#                     errorboxes = []
#                     points = set()
#                     sparsity = []

#                     for reg in regs:
#                         point_results = data[(data['Depth Limit'] == depth)&(data['Regularization'] == reg)]
#                         sparsity.append(np.median(point_results['# Leaves']))
#                         point_results.loc[point_results[x_axis] > 1800, x_axis] = 1800
#                         x_iqr = point_results[x_axis].quantile([0.25, 0.5, 0.75])
#                         y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])

#                         point = (
#                             x_iqr[0.5], y_iqr[0.5],
#                             x_iqr[0.5] - x_iqr[0.25], x_iqr[0.75] - x_iqr[0.5],
#                             y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])

#                         if not point in points:
#                             points.add(point)
#                             x.append(point[0]); y.append(point[1])
#                             x_low.append(point[2]); x_high.append(point[3])
#                             y_low.append(point[4]); y_high.append(point[5])

#                     for xv, yv, xl, xh, yl, yh in zip(x, y, x_low, x_high, y_low, y_high):
#                         rect = Rectangle((xv - xl, yv - yl), max(xl + xh, 0.00001), max(yl + yh, 0.00001))
#                         errorboxes.append(rect)
#                     if algorithm == 'dl85':
#                         colors = '#e3a042'
#                     elif algorithm == 'gosdt' or algorithm == 'gosdt_warm_lb':
#                         colors = '#509eba'
#                     if k == 0:
#                         markers = '.'
#                     elif k == 1:
#                         markers ='*'

#                     for index in range(len(x)):
#                         axs[j].errorbar(x[index], y[index], xerr=np.array([x_low[index], x_high[index]]).reshape(2,1), yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
#                                      markersize=sparsity[index]+10, marker=markers,
#                                      color=colors, alpha=1,
#                                      linewidth=1, linestyle='none')


#             axs[j].set_xscale('log')
#             axs[j].set_xlabel('Training Time (s)')
#             axs[j].set_ylabel(y_axis + " (%)")
#             axs[j].axhline(ref_acc, color='black', ls='--')
#             if depth == -1:
#                 axs[j].set_title("{} vs Run Time\n{} (no depth constraint)".format(y_axis, title))
#             else:
#                 axs[j].set_title("{} vs Run Time\n{} (depth={})".format(y_axis, title, depth))

#         handles = []
#         for c in ['#e3a042', '#509eba']:
#             for m in ['.','*']:
#                 line = Line2D([], [], color=c, marker=m, markersize=18, ls='none')
#                 barline = LineCollection(np.empty((1,2,2)), colors=c, linewidths=1)
#                 handles.append(ErrorbarContainer((line, (), [barline]), has_xerr=True, has_yerr=True))
#         handles.append(Line2D([], [], color='black', ls='--'))
#         labels = ['dl8.5 w/ guessing', 'dl8.5 w/o guessing', 'gosdt w/ guessing', 'gosdt w/o guessing', 'GBDT']
#         axs[j].legend(handles, labels, prop={'size': 14})#,bbox_to_anchor=(1, 1)  )

#         plt.savefig("{}_{}_vs_runtime_{}.png".format(figurename, y_axis.lower().replace(' ', '_'), title), bbox_inches='tight')
#         plt.savefig("{}_{}_vs_runtime_{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), title), bbox_inches='tight')



# # Figure 4
# def plot_bar_guessing_thresh(figurename, excelpath, algorithm, d, y_axis):
#     datasets = ['spiral', '2y_score', 'general_2y', 'carryout', 'rest20', 'fico', 'dataGeneral']
#     regs = [0.01, 0.001, 0.005, 0.001, 0.001, 0.0005, 0.0002]
#     points_thresh = []
#     points_no_thresh = []
#     with open(CONFIG_REFERENCE_ACC, 'r') as f:
#         ref = json.loads(f.read())
#     title = [ref[dataset]['title'] for dataset in datasets]
#     title = [title[i] + '\nλ='+str(regs[i]) for i in range(len(datasets))]
#     if y_axis == 'Test Accuracy':
#         plt.figure(figsize=(7.5,4))
#         datasets = datasets[1:]
#         title = title[1:]
#         regs = regs[1:]
#     else:
#         plt.figure(figsize=(8.5,4))
#     index = np.arange(len(datasets))
#     width=0.4
#     for k, dataset in enumerate(datasets):
#         reg = regs[k]
#         df_thresh = pd.read_excel(excelpath, dataset+'_thresh')
#         thresh = df_thresh[(df_thresh['Algorithm'] == algorithm)&(df_thresh['Depth Limit'] == d)&(df_thresh['Regularization'] == reg)]

#         df_no_thresh = pd.read_excel(excelpath, dataset+'_no_thresh')
#         no_thresh = df_no_thresh[(df_no_thresh['Algorithm'] == algorithm)&(df_no_thresh['Depth Limit'] == d)&(df_no_thresh['Regularization'] == reg)]

#         if y_axis=='Training Time':
#             if (no_thresh[y_axis] == 'None').any():
#                 points_no_thresh.append(-1)

#             else:
#                 no_thresh.loc[no_thresh[y_axis] > 1800, y_axis] = 1800
#                 points_no_thresh.append(np.mean(no_thresh[y_axis]))

#             thresh.loc[thresh[y_axis] > 1800, y_axis] = 1800
#             points_thresh.append(np.mean(thresh[y_axis]))


#         else:
#             if (no_thresh[y_axis] == 'None').any():
#                 points_no_thresh.append(0.5)
#             else:
#                 points_no_thresh.append(np.mean(no_thresh[y_axis]))

#             if (thresh[y_axis] == 'None').any():
#                 points_thresh.append(0.5)
#             else:
#                 points_thresh.append(np.mean(thresh[y_axis]))

#     points_thresh = np.array(points_thresh)
#     points_no_thresh = np.array(points_no_thresh)

#     if y_axis == 'Training Time':
#         plt.bar(index - width/2, points_thresh, width, bottom=0, color='#4c72b0', label='Training time w guessing')
#         #plt.bar(index - width/2, points_thresh[:,1], width, bottom=points_thresh[:,0], color='#4c72b0', alpha=0.7, label='Guess time')
#         i = np.where(points_no_thresh == -1)[0] # where memory out
#         j = np.where(points_no_thresh == 1800)[0]
#         k = np.where((points_no_thresh != -1) & (points_no_thresh < 1800))[0]

#         plt.bar(index[k] + width/2, points_no_thresh[k], width, bottom=0, color='#dd8452',  label='Training time w/o guessing')
#         plt.bar(index[j] + width/2, points_no_thresh[j], width, bottom=0, color='#dd8452', hatch='/', label='Time Out')
#         plt.bar(index[i] + width/2, np.repeat(1800, len(i)), width, bottom=0, color='silver', hatch='/', label='Memory out')

#         plt.ylabel('Time (s)', fontsize=20)


#         colors = ['#4c72b0', '#dd8452']
#         labels = ['w guessing', 'w/o guessing']
#         hatchs = [None, None]
#         if len(j) > 0:
#             colors.append('#dd8452')
#             labels.append('Time out')
#             hatchs.append(r'///')
#         if len(i) > 0:
#             colors.append('silver')
#             labels.append('Memory out')
#             hatchs.append(r'///')

#         handles = [mpatches.Patch(facecolor=colors[i], hatch = hatchs[i]) for i in range(len(colors))]
#         plt.legend(handles, labels, prop={'size': 14})

#     elif y_axis == 'Training Accuracy' or y_axis=='Test Accuracy':
#         i = np.where(points_no_thresh == 0.5)[0]
#         plt.bar(index - width/2, points_thresh, width, bottom=0, color='#ccb974', label='w guessing')
#         plt.bar(np.delete(index, i) + width/2, np.delete(points_no_thresh, i), width, bottom=0, color='#8172b3', label='w/o guessing')
#         plt.bar(index[i] + width/2, points_no_thresh[i], width, bottom=0, color='#8172b3', hatch='/', label='N/A')

#         plt.ylabel(y_axis, fontsize=20)

#         colors = ['#ccb974', '#8172b3']
#         labels = ['w guessing', 'w/o guessing']
#         hatchs = [None, None]
#         if len(i) > 0:
#             colors.append('#8172b3')
#             labels.append('N/A')
#             hatchs.append(r'///')
#         handles = [mpatches.Patch(facecolor=colors[i], hatch = hatchs[i]) for i in range(len(colors))]
#         plt.legend(handles, labels, prop={'size': 14})

#     plt.yticks(fontsize=15)
#     plt.xticks(index, title, fontsize=18, rotation=-90)
#     if d != -1:
#         plt.title('{} {} \ndepth={}'.format(algorithm, y_axis.lower(), d), fontsize=20)
#     else:
#         plt.title('{} {} \nno depth constraint'.format(algorithm, y_axis.lower()), fontsize=20)
#     plt.savefig("{}_threshold_{}_{}_d{}.png".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(d)), bbox_inches='tight')
#     plt.savefig("{}_threshold_{}_{}_d{}.pdf".format(figurename, algorithm, y_axis.lower().replace(' ', '_'), str(d)), bbox_inches='tight')


# # Figure 7
# def plot_bar_guessing_lb(figurename, excelpath, d, y_axis):
#     datasets = ['2y_score', 'general_2y', 'carryout', 'rest20', 'fico', 'dataGeneral']
#     regs = [0.001, 0.005, 0.001, 0.001, 0.0005, 0.0002]
#     points_guess = []
#     points_no_guess = []
#     with open(CONFIG_REFERENCE_ACC, 'r') as f:
#         ref = json.loads(f.read())
#     title = [ref[dataset]['title'] for dataset in datasets]
#     title = [title[i] + '\nλ='+str(regs[i]) for i in range(len(datasets))]
#     plt.figure(figsize=(7.5,4))
#     index = np.arange(len(datasets))
#     width=0.4
#     for k, dataset in enumerate(datasets):
#         reg = regs[k]
#         df_guess = pd.read_excel(excelpath, dataset+'_thresh')
#         guess = df_guess[(df_guess['Algorithm'] == 'gosdt_warm_lb')&(df_guess['Depth Limit'] == d)&(df_guess['Regularization'] == reg)]
#         no_guess = df_guess[(df_guess['Algorithm'] == 'gosdt')&(df_guess['Depth Limit'] == d)&(df_guess['Regularization'] == reg)]

#         if y_axis=='Training Time':
#             no_guess.loc[no_guess[y_axis] > 1800, y_axis] = 1800
#             points_no_guess.append(np.mean(no_guess[y_axis]))

#             guess.loc[guess[y_axis] > 1800, y_axis] = 1800
#             points_guess.append(np.mean(guess[y_axis]))


#         else:
#             points_no_guess.append(np.mean(no_guess[y_axis]))
#             points_guess.append(np.mean(guess[y_axis]))

#     points_guess = np.array(points_guess)
#     points_no_guess = np.array(points_no_guess)

#     if y_axis == 'Training Time':
#         plt.bar(index - width/2, points_guess, width, bottom=0, color='#4c72b0')
#         i = np.where(points_no_guess == -1)[0] # where memory out
#         j = np.where(points_no_guess == 1800)[0]
#         k = np.where((points_no_guess != -1) & (points_no_guess < 1800))[0]

#         plt.bar(index[k] + width/2, points_no_guess[k], width, bottom=0, color='#dd8452')
#         plt.bar(index[j] + width/2, points_no_guess[j], width, bottom=0, color='#dd8452', hatch='/')
#         plt.bar(index[i] + width/2, np.repeat(1800, len(i)), width, bottom=0, color='silver', hatch='/')

#         plt.ylabel('Time (s)', fontsize=20)

#         colors = ['#4c72b0', '#dd8452']
#         labels = ['w guessing', 'w/o guessing']
#         hatchs = [None, None]
#         if len(j) > 0:
#             colors.append('#dd8452')
#             labels.append('Time out')
#             hatchs.append(r'///')
#         if len(i) > 0:
#             colors.append('silver')
#             labels.append('Memory out')
#             hatchs.append(r'///')

#         handles = [mpatches.Patch(facecolor=colors[i], hatch = hatchs[i]) for i in range(len(colors))]
#         plt.legend(handles, labels, prop={'size': 14})

#     elif y_axis == 'Training Accuracy' or y_axis=='Test Accuracy':
#         i = np.where(points_no_guess == 0.5)[0]
#         plt.bar(index - width/2, points_guess, width, bottom=0, color='#ccb974')
#         plt.bar(np.delete(index, i) + width/2, np.delete(points_no_guess, i), width, bottom=0, color='#8172b3')
#         plt.bar(index[i] + width/2, points_no_guess[i], width, bottom=0, color='#8172b3', hatch='/')

#         plt.ylabel(y_axis, fontsize=20)

#         colors = ['#ccb974', '#8172b3']
#         labels = ['w guessing', 'w/o guessing']
#         hatchs = [None, None]
#         if len(i) > 0:
#             colors.append('#8172b3')
#             labels.append('N/A')
#             hatchs.append(r'///')
#         handles = [mpatches.Patch(facecolor=colors[i], hatch = hatchs[i]) for i in range(len(colors))]
#         plt.legend(handles, labels, prop={'size': 14})

#     plt.yticks(fontsize=15)
#     plt.xticks(index, title, fontsize=18, rotation=-90)
#     if d != -1:
#         plt.title('gosdt {} \ndepth={}'.format(y_axis.lower(), d), fontsize=20)
#     else:
#         plt.title('gosdt {} \nno depth constraint'.format(y_axis.lower()), fontsize=20)
#     plt.savefig("{}_gosdt_lb_{}_d{}.png".format(figurename, y_axis.lower().replace(' ', '_'), str(d)), bbox_inches='tight')
#     plt.savefig("{}_gosdt_lb_{}_d{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), str(d)), bbox_inches='tight')


# # Figure 6
# def plot_depth_guess(figurename, excelpath, dataset, reg):
#     algorithm = 'gosdt'
#     df = pd.read_excel(excelpath, dataset+'_thresh')
#     with open(CONFIG_REFERENCE_ACC, 'r') as f:
#         ref = json.loads(f.read())
#     for y_axis in ['Training Time', 'Accuracy']:
#         plt.rc('font', size = 18)

#         title = ref[dataset]['title']

#         x_axis = "Depth Limit"
#         plt.figure(figsize=(9,5))

#         df = df[df['Algorithm'] == algorithm]

#         if y_axis == 'Accuracy':
#             z=[]; z_low = []; z_high = [] # store test accuracy
#         y=[]; y_low = []; y_high = []
#         depth_off = [-2, -1, 0, 1, 2]
#         data = df[(df['Regularization']==reg)]
#         depth_opt = data['Optimal Depth'].values[0]-1
#         print(depth_opt)
#         for off in depth_off:
#             depth = depth_opt+off
#             point_results = data[(data['Depth Limit'] == depth)]
#             if y_axis == 'Accuracy':
#                 y_iqr = point_results['Training Accuracy'].quantile([0.25, 0.5, 0.75])
#                 z_iqr = point_results['Test Accuracy'].quantile([0.25, 0.5, 0.75])
#                 z.append(z_iqr[0.5]); z_low.append(z_iqr[0.5]-z_iqr[0.25]);
#                 z_high.append(z_iqr[0.75]-z_iqr[0.5])
#             else:
#                 y_iqr = point_results[y_axis].quantile([0.25, 0.5, 0.75])
#             point = (y_iqr[0.5],
#                 y_iqr[0.5] - y_iqr[0.25], y_iqr[0.75] - y_iqr[0.5])
#             print(off, point)

#             y.append(point[0])
#             y_low.append(point[1]); y_high.append(point[2])



#         for index in range(len(depth_off)):
#             if depth_off[index] == 0:
#                 c = 'C1'
#             else:
#                 c = 'C0'
#             plt.errorbar(depth_off[index], y[index],
#                          yerr=np.array([y_low[index], y_high[index]]).reshape(2,1),
#                          markersize=15, marker='o',
#                          color=c, alpha=1,
#                          linewidth=1, linestyle='none')
#             if y_axis == 'Accuracy':
#                 plt.errorbar(depth_off[index], z[index],
#                          yerr=np.array([z_low[index], z_high[index]]).reshape(2,1),
#                          markersize=15, marker='^',
#                          color=c, alpha=1,
#                          linewidth=1, linestyle='none')
#         plt.grid(color='0.93')
#         plt.xticks(depth_off)
#         plt.xlabel('Offset from optimal depth={}'.format(depth_opt))
#         plt.ylabel(y_axis)
#         plt.title("{} vs Depth Limit\n{}".format(y_axis, title))

#         handles = []
#         for c in ['C1', 'C0']:
#             line = Line2D([], [], color=c, marker='o', markersize=10, ls='none')
#             handles.append(line)
#             if y_axis == 'Accuracy':
#                 line = Line2D([], [], color=c, marker='^', markersize=10, ls='none')
#                 handles.append(line)
#         if y_axis == 'Accuracy':
#             labels = ['training accuracy at optimal depth', 'test accuracy at optimal depth',
#                       'training accuracy at suboptimal depth', 'test accuracy at suboptimal depth']
#         else:
#             labels = ['optimal depth', 'suboptimal depth']
#         plt.legend(handles, labels, prop={'size': 14})#,bbox_to_anchor=(1, 1)  )

#         plt.savefig("{}_depth_guess_{}_vs_{}_{}.png".format(figurename, y_axis.lower().replace(' ', '_'), x_axis.lower().replace(' ','_'), title), bbox_inches='tight')
#         plt.savefig("{}_depth_guess_{}_vs_{}_{}.pdf".format(figurename, y_axis.lower().replace(' ', '_'), x_axis.lower().replace(' ','_'), title), bbox_inches='tight')
