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
        5: 'RR1',       6:'RR2',
        7: 'RRE',
        21:'Hawkins',

        24:'VfNc',


        27:'SampleLam',

        37:'BLam0.1',
        38:'BLam0.2',
        39:'BLam0.3',
        40:'BLam0.5'
    }







def sensitivityAnalysisPlot(savestring, root, firstseed, num_trials, ignore_missing=False):
    



    # savestring = 'sl_sa_2_19'
    n_list = [250]
    budget_list = [0.1]
    n_states_list = [4]
    percent_random_list = np.concatenate([np.linspace(0, 0.5, 11), np.linspace(0.55, 1.0, 10)])
    percent_random_list = percent_random_list[:17]

    

    # policies_to_plot = [21, 30, 27, 24, 0]
    policies_to_plot = [21, 27, 0]


    plot_data = np.zeros((len(policies_to_plot), len(percent_random_list)))
    errors = np.zeros((len(policies_to_plot), len(percent_random_list)))


    # datatype = 'healthcare'
    datatype = 'sl_sa'

    budget_frac = budget_list[0]
    state_size = n_states_list[0]
    prob_size = n_list[0]

    L = 40


    for i,p in enumerate(policies_to_plot):
        for x, pl0 in enumerate(percent_random_list):
            rewards = []
            for j in range(num_trials):
                seed = firstseed + j
                file_template = 'logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s.npy'

                filename = file_template%(savestring, prob_size, budget_frac, L, p, datatype, seed, state_size, pl0)

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


    adjusted_plot_data = np.copy(plot_data)
    adjusted_errors = np.copy(errors)
    
    nobody_ind = -1
    for i,p in enumerate(policies_to_plot):
        for x, pl0 in enumerate(percent_random_list):
            
            if p != 0:
                adjusted_plot_data[i,x] -= adjusted_plot_data[nobody_ind,x]
                adjusted_errors[i,x] = np.sqrt(adjusted_errors[i,x]**2 + adjusted_errors[nobody_ind,x]**2)
    policies_to_plot.remove(0)

    plot_data = adjusted_plot_data
    errors = adjusted_errors

    linestyles = ['--', '-.', '-', ':']
    fig, ax = plt.subplots(figsize=(8,6))
    
    for i,p in enumerate(policies_to_plot):
        if pname[p] =='Hawkins':
            ax.errorbar(percent_random_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle='--')#linestyles[i%4], capsize=5)
        elif pname[p] in ['VfNc','nobody']:
            ax.errorbar(percent_random_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle=':')#linestyles[i%4], capsize=5)
        elif pname[p] in ['SampleLam']:
            ax.errorbar(percent_random_list, plot_data[i], yerr=errors[i], label=pname[p], linestyle='-.')#linestyles[i%4], capsize=5)
        else:
            ax.errorbar(percent_random_list, plot_data[i], yerr=errors[i], label=pname[p])#, linestyle=linestyles[i%4], capsize=5)
    
    # ax.set_ylim([0,3000])
    
    ax.set_ylabel('Discounted Sum of Rewards')
    
    ax.set_xlabel(r'$\alpha$')
    # ax.set_title('Runtimes on %s'%datatype, fontsize=14)   
    # ax.legend(loc='upper left')
    
    # ax.legend(bbox_to_anchor =(0.2, .8),loc='center')
    ax.legend(ncol=4, bbox_to_anchor =(0.48, 1.15),loc='center')
       
    plt.tight_layout() 

    outname = '%s_b%.1f_s%s_N%s_sensitivity_analysis.pdf'%(savestring,budget_frac,state_size,prob_size)

    plt.savefig(outname)
    plt.show()









savestring = 'sl_sa_2_19'


root='./../batches/%s'%savestring
firstseed=0
num_trials=20


sensitivityAnalysisPlot(savestring, root, firstseed, num_trials, ignore_missing=True)


