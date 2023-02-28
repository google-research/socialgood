import matplotlib
# matplotlib.use('pdf')

import numpy as np 
import matplotlib.pyplot as plt
import os
import pandas as pd
import sys
import glob
from tqdm import tqdm
from matplotlib.lines import Line2D
from matplotlib import rcParams
from numba import jit 

#Ensure type 1 fonts are used
import matplotlib as mpl
mpl.rcParams['ps.useafm'] = True
mpl.rcParams['pdf.use14corefonts'] = True
mpl.rcParams['text.usetex'] = True
params = {'text.latex.preamble' : [r'\usepackage{bm}']}
plt.rcParams.update(params)
# mpl.rcParams['text.latex.unicode']=True

SMALL_SIZE = 16
MEDIUM_SIZE = 24
BIGGER_SIZE = 32
plt.rc('font', weight='bold')
plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
plt.rc('xtick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('ytick', labelsize=MEDIUM_SIZE)    # fontsize of the tick labels
plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title

# policy names dict
pname={
        0: 'nobody',    2: 'Random',
        3: 'Myopic',    4: 'MDPoptimal', 
        5: 'RR1',       6:'RR2',
        7: 'RRE',
        19:'La*',       20:'La1', 
        21:'Hawkins',
        22:'La*H',      23:'La1H',

        24:'VfNc',

        25:'IterLam',
        26:'RandLam',

        27:'SampleLam',
        28:'QLam',
        30:'Blam0.2',
        31:'Blam0.5',
        32:'Blam1.0',

        33:'BL+SL0.1',
        34:'BL+SL0.2',
        35:'BL+SL0.3',
        36:'BL+SL0.5',

        37:'BLam0.1',
        38:'BLam0.2',
        39:'BLam0.3',
        40:'BLam0.5'
    }

def runtimePlot(savestring, root, firstseed, num_trials, bfi, ssi, ignore_missing=False, logplot=False):
    

    # savestring = 'healthcare_9_26'
    # n_list = [100, 200, 300, 500]
    # budget_frac_list = [0.1, 0.2, 0.5]
    # state_size_list = [3, 6, 9]

    # savestring = 'healthcare_10_1'
    # n_list = [100, 250, 500, 750]
    # budget_frac_list = [0.1, 0.2, 0.5]
    # state_size_list = [3, 4, 5]


    # savestring = 'eng11_trials_resubmit'
    n_list = [100, 200, 500, 750]
    percent_lam0_list = [0.5, 0.8]
    state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]

    

    # policies_to_plot = [21, 30, 27, 24, 0]
    policies_to_plot = [21, 30, 31, 32, 27, 24, 0]

    # policies_to_plot = [21, 38, 39, 40, 27, 24, 0]
    # policies_to_plot = [34, 38, 35, 39, 36, 40]
    # policies_to_plot = [34, 38]

    # problem_sizes = [50, 200, 500, 2000]
    # problem_sizes = [300, 600, 900]
    # problem_sizes = [50, 100, 200]

    plot_data = np.zeros((len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(policies_to_plot), len(n_list)))


    # datatype = 'healthcare'
    datatype = 'eng11'

    budget_frac = 0.5
    pl0 = percent_lam0_list[bfi]
    # pl0=0
    # budget_frac = budget_frac_list[bfi]
    state_size = state_size_list[ssi]
    L = 40


    for i,p in enumerate(policies_to_plot):
        for x, prob_size in enumerate(n_list):
            run_times = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/runtime/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, pl0)

                fname = os.path.join(root,filename)

                try:
                    run_times.append(np.load(fname))
                    # print(run_times)
                # if we hit the walltime
                except Exception as e:
                    if not ignore_missing:
                        run_times.append(60*60*4) # seconds, minutes, hours
                    print(e)
                    pass

            run_times = np.array(run_times)
            plot_data[i,x] = run_times.mean()
            errors[i,x] = run_times.std()
            print(run_times)


    linestyles = ['--', '-.', '-', ':']
    fig, ax = plt.subplots(figsize=(8,6))
    for i,p in enumerate(policies_to_plot):
        if pname[p] =='Hawkins':
            ax.errorbar(n_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle='--')#linestyles[i%4], capsize=5)
        elif pname[p] in ['VfNc','nobody']:
            ax.errorbar(n_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle=':')#linestyles[i%4], capsize=5)
        elif pname[p] in ['SampleLam']:
            ax.errorbar(n_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle='-.')#linestyles[i%4], capsize=5)
        else:
            ax.errorbar(n_list, plot_data[i], yerr=errors[i], label=pname[p])#, linestyle=linestyles[i%4], capsize=5)
    
    # ax.set_ylim([0,3000])
    
    if not logplot:
        ax.set_ylabel('Runtime (s)')
    else:
        ax.set_ylabel(r'Runtime $\bm{\log_{10}}$(s)')
        ax.set_yscale('log', base=10)


    ax.set_xlabel('Number of Arms (N)')
    # ax.set_title('Runtimes on %s'%datatype, fontsize=14)   
    # ax.legend(loc='upper left')
    
    # ax.legend(bbox_to_anchor =(0.2, .8),loc='center')
    ax.legend(ncol=4, bbox_to_anchor =(0.48, 1.15),loc='center')
       
    plt.tight_layout() 

    if not logplot:
        outname = '%s_b%.1f_s%s_pl0%s_runtime.png'%(savestring,budget_frac,state_size,pl0)
    else:
        outname = '%s_b%.1f_s%s_pl0%s_runtime_logplot.png'%(savestring,budget_frac,state_size,pl0)
    # outname = '%s_b%.1f_s%s_runtime.png'%(savestring,budget_frac,state_size)
    # outname = 'debug_%s_b%.1f_s%s_runtime.png'%(savestring,budget_frac,state_size)

    plt.savefig(outname)
    plt.show()











def groupedBarPlot(savestring, root, firstseed, num_trials, bfi, ssi):
    

    # # savestring = 'healthcare_9_26'
    # n_list = [100, 200, 300, 500]
    # budget_frac_list = [0.1, 0.2, 0.5]
    # state_size_list = [3, 6, 9]

    # savestring = 'healthcare_10_1'
    n_list = [100, 250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]

    # savestring = 'eng11_trials_resubmit'
    # n_list = [100, 200, 500, 750]
    # percent_lam0_list = [0.5, 0.8]
    # state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]


    # policies_to_plot = [0, 5, 6, 7, 24, 27, 21, 30, 31, 32]
    # policies_to_plot = [0, 5, 24, 27, 21, 30, 31, 32]
    policies_to_plot = [0, 24, 27, 21, 38, 39, 40]
    # policies_to_plot = [0, 24, 27, 21, 38, 34, 35, 36]
    # policies_to_plot = [33, 37, 34, 38, 35, 39, 36, 40]


    # problem_sizes = [50, 200, 500, 2000]
    # problem_sizes = [300, 600, 900]
    # problem_sizes = [50, 100, 200]

    plot_data = np.zeros((len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(policies_to_plot), len(n_list)))


    datatype = 'healthcare'
    # datatype = 'eng11'
    L = 40

    x_axis_label = 'Number of arms'
    # x_axis_label = 'Budget (as fraction of N)'
    # x_axis_label = 'Number of states'

    problem_size = 0
    budget_frac = 0
    state_size = 0
    pl0 = 0

    if x_axis_label == 'Number of arms':
        x_axis = n_list
        budget_frac = budget_frac_list[bfi]
        state_size = state_size_list[ssi]
        # pl0 = percent_lam0_list[bfi]

        # title = 'Varying number patients, %s budget per day; trials: %s; States: %s ' % (N, B, N_TRIALS, args.num_states)

    elif x_axis_label == 'Budget (as fraction of N)':
        problem_size = n_list[3]
        x_axis = budget_frac_list
        state_size = state_size_list[1]

    elif x_axis_label == 'Number of states':
        problem_size = n_list[3]
        budget_frac - budget_frac_list[1]
        x_axis = state_size_list




    for i,p in enumerate(policies_to_plot):
        for x, detail in enumerate(x_axis):
            rewards = []
            for j in range(num_trials):
                seed = firstseed + j

                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s.npy'

                if x_axis_label == 'Number of arms':
                    filename = file_template%(savestring, detail, budget_frac, L, p, datatype, seed, state_size, pl0)

                elif x_axis_label == 'Budget (as fraction of N)':
                    filename = file_template%(savestring, problem_size, detail, L, p, datatype, seed, state_size)

                elif x_axis_label == 'Number of states':
                    filename = file_template%(savestring, problem_size, budget_frac, L, p, datatype, seed, detail)

                fname = os.path.join(root,filename)

                try:
                    reward = np.load(fname)
                    reward = reward.sum()/reward.shape[0] # divide by N
                    rewards.append(reward)

                except Exception as e:
                    print(e)
                    pass

            rewards = np.array(rewards)
            plot_data[i,x] = rewards.mean()
            errors[i,x] = rewards.std()


    labels = x_axis


    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.1  # the width of the bars
    
    colors = [
    # '#0000ff',
    # '#1919ff',
    '#3232ff',
    # '#4c4cff',
    '#6666ff',
    # '#7f7fff',
    # '#9999ff',
    '#b2b2ff',
    # '#ccccff',
    # '#e5e5ff',
    '#ffffff'
    ][::-1]

    patterns = [ "/" , "."]

    rects = []
    fig, ax = plt.subplots(figsize=(9,5))
    for i,p in enumerate(policies_to_plot):
        rects1 = ax.bar(x+i*width, plot_data[i], width, yerr=errors[i], label=pname[p], 
            color=colors[i%4], edgecolor='k', linewidth='0.5', hatch=patterns[i//4], capsize=4)
        rects.append(rects1)

    ymin = plot_data[np.logical_not(np.isnan(plot_data))].min() - 1
    ymax = plot_data[np.logical_not(np.isnan(plot_data))].max() + 0.5

    ax.set_ylim([ymin,ymax])

    ylabel = 'Discounted sum of rewards'
    ax.set_ylabel(ylabel)
    ax.set_xlabel(x_axis_label)
    # ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    # ax.legend(ncol=len(policies_to_plot)//2, bbox_to_anchor =(.95, 1.35)) # healthcare
    ax.legend(ncol=4, bbox_to_anchor =(0.5, 1.3),loc='center') # eng11
    # ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
    # for r in rects:
    #     autolabel(r)       
    plt.tight_layout() 

    # outname = '%s_b%.1f_s%s_pl0%s_rewards.png'%(savestring,budget_frac,state_size,pl0)
    outname = '%s_b%.1f_s%s_rewards.png'%(savestring,budget_frac,state_size)
    # outname = 'debug_%s_b%.1f_s%s_rewards.png'%(savestring,budget_frac,state_size)
    plt.savefig(outname)
    # plt.show()



# savestring = 'init_runtime'
# savestring = 'both_beat_lam0'
# savestring = 'blam_better_runtimes'
# savestring = 'fixed_memb_bug_cannon_trials'
# savestring = 'heatlhcare_trial_more_good_patients'
# savestring = 'eng11_trials'
# savestring = 'heatlhcare_trial_less_good_patients'
# savestring = 'heatlhcare_MANY_good_patients'
savestring = 'eng11_trials_resubmit'

# savestring = 'healthcare_9_26'
# savestring = 'healthcare_10_1'

root='./../batches/%s'%savestring
firstseed=0
num_trials=25


# for i in range(3):
#     for j in range(3):
#         runtimePlot(savestring, root, firstseed, num_trials, i, j)
#         groupedBarPlot(savestring, root, firstseed, num_trials, i, j)

i=0
j=2
runtimePlot(savestring, root, firstseed, num_trials, i, j, ignore_missing=True, logplot=False)
# groupedBarPlot(savestring, root, firstseed, num_trials, i, j)

# i=0
# j=2
# runtimePlot(savestring, root, firstseed, num_trials, i, j, ignore_missing=True)
# groupedBarPlot(savestring, root, firstseed, num_trials, i, j)



