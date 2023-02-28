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
MEDIUM_SIZE = 16
BIGGER_SIZE = 16
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

        25:'IterLam',
        26:'RandLam',

        27:'SampleLam',

        37:'BLam0.1',
        38:'BLam0.2',
        39:'BLam0.3',
        40:'BLam0.5'
    }

def runtimePlotLoopStates(savestring, root, firstseed, num_trials, bfi, ignore_missing=False, logplot=False):
    


    # savestring = 'healthcare_2_19'
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]


    # savestring = 'eng11_trials_resubmit'
    # n_list = [100, 200, 500, 750]
    # percent_lam0_list = [0.5, 0.8]
    # state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]

    

    policies_to_plot = [21, 38, 39, 40, 27, 24, 0]

    plot_data = np.zeros((len(state_size_list), len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(state_size_list), len(policies_to_plot), len(n_list)))


    datatype = 'healthcare'
    # datatype = 'eng11'

    # budget_frac = 0.5
    # pl0 = percent_lam0_list[bfi]
    pl0=0
    budget_frac = budget_frac_list[bfi]
    # state_size = state_size_list[ssi]
    L = 40

    for k in range(len(state_size_list)):
        state_size = state_size_list[k]
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
                plot_data[k,i,x] = run_times.mean()
                errors[k,i,x] = run_times.std()
                print(run_times)


    linestyles = ['--', '-.', '-', ':']
    fig, ax = plt.subplots(1,3, figsize=(14,3.5))
    for k,s in enumerate(state_size_list):
        for i,p in enumerate(policies_to_plot):
            if pname[p] =='Hawkins':
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle='--')#linestyles[i%4], capsize=5)
            elif pname[p] in ['VfNc','nobody']:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle=':')#linestyles[i%4], capsize=5)
            elif pname[p] in ['SampleLam']:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle='-.')#linestyles[i%4], capsize=5)
            else:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p])#, linestyle=linestyles[i%4], capsize=5)
    
    # ax.set_ylim([0,3000])
    
    lsize = 16
    if not logplot:
        ax[0].set_ylabel('Runtime (s)')
        ax[0].set_xlabel('(d)')
        ax[1].set_xlabel('(e)')
        ax[2].set_xlabel('(f)')
    else:
        ax[0].set_ylabel(r'Runtime ($\bm{\log_{10}}$(s))')
        ax[0].set_xlabel('(g)',fontsize=lsize)
        ax[1].set_xlabel('(h)',fontsize=lsize)
        ax[2].set_xlabel('(i)',fontsize=lsize)
        for axis in ax:
            axis.set_yscale('log', base=10)

    # fig.set_xlabel('Number of Arms (N)')
    fig.text(0.5, 0.03, 'Number of Arms (N)', ha='center')

    # ax.set_title('Runtimes on %s'%datatype, fontsize=14)   
    # ax.legend(loc='upper left')
    ax[1].legend(ncol=7, bbox_to_anchor =(0.45, 1.12),loc='center')
       
    plt.tight_layout() 
    plt.subplots_adjust(
                    wspace=0.3, 
                    hspace=0.0)
    # plt.title('B:%s'%budget_frac)


    if not logplot:
        outname = '%s_b%.1f_pl0%s_stateloop_runtime.pdf'%(savestring,budget_frac,pl0)
    else:
        outname = '%s_b%.1f_pl0%s_stateloop_runtime_logplot.pdf'%(savestring,budget_frac,pl0)


    plt.savefig(outname)
    plt.show()




def runtimePlotLoopBudgets(savestring, root, firstseed, num_trials, ssi, ignore_missing=False, logplot=False):
    


    # savestring = 'healthcare_10_1'
    # n_list = [100, 250, 500, 750]
    # budget_frac_list = [0.1, 0.2, 0.5]
    # state_size_list = [3, 4, 5]

    # savestring = 'healthcare_2_19'
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]


    # savestring = 'eng11_trials_resubmit'
    # n_list = [100, 200, 500, 750]
    # percent_lam0_list = [0.5, 0.8]
    # state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]

    policies_to_plot = [21, 38, 39, 40, 27, 24, 0]
    

    plot_data = np.zeros((len(budget_frac_list), len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(budget_frac_list), len(policies_to_plot), len(n_list)))


    datatype = 'healthcare'
    # datatype = 'eng11'

    # budget_frac = 0.5
    # pl0 = percent_lam0_list[bfi]
    pl0=0
    # budget_frac = budget_frac_list[bfi]
    state_size = state_size_list[ssi]
    L = 40

    for k in range(len(budget_frac_list)):
        budget_frac = budget_frac_list[k]
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
                plot_data[k,i,x] = run_times.mean()
                errors[k,i,x] = run_times.std()
                print(run_times)


    linestyles = ['--', '-.', '-', ':']
    fig, ax = plt.subplots(1,3, figsize=(14,3.5))
    for k,s in enumerate(budget_frac_list):
        for i,p in enumerate(policies_to_plot):
            if pname[p] =='Hawkins':
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle='--')#linestyles[i%4], capsize=5)
            elif pname[p] in ['VfNc','nobody']:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle=':')#linestyles[i%4], capsize=5)
            elif pname[p] in ['SampleLam']:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p], linestyle='-.')#linestyles[i%4], capsize=5)
            else:
                ax[k].errorbar(n_list, plot_data[k,i], yerr=errors[k,i], label=pname[p])#, linestyle=linestyles[i%4], capsize=5)
    
    # ax.set_ylim([0,3000])
    
    if not logplot:
        ax[0].set_ylabel('Runtime (s)')
        lsize = 16
        ax[0].set_xlabel('(d)',fontsize=lsize)
        ax[1].set_xlabel('(e)',fontsize=lsize)
        ax[2].set_xlabel('(f)',fontsize=lsize)
    else:
        ax[0].set_ylabel(r'Runtime ($\bm{\log_{10}}$(s))')
        lsize = 16
        ax[0].set_xlabel('(g)',fontsize=lsize)
        ax[1].set_xlabel('(h)',fontsize=lsize)
        ax[2].set_xlabel('(i)',fontsize=lsize)
        for axis in ax:
            axis.set_yscale('log', base=10)


    # fig.set_xlabel('Number of Arms (N)')
    fig.text(0.5, 0.03, 'Number of Arms (N)', ha='center',fontsize=lsize)

    # ax.set_title('Runtimes on %s'%datatype, fontsize=14)   
    # ax.legend(loc='upper left')
    ax[1].legend(ncol=7, bbox_to_anchor =(0.45, 1.15),loc='center')
       
    plt.tight_layout() 
    plt.subplots_adjust(
                    wspace=0.3, 
                    hspace=0.0)

    # plt.title('S:%s'%state_size)

    if not logplot:
        outname = '%s_s%s_pl0%s_budgetloop_runtime.pdf'%(savestring,state_size,pl0)
    else:
        outname = '%s_s%s_pl0%s_budgetloop_runtime_logscale.pdf'%(savestring,state_size,pl0)

    plt.savefig(outname)
    # plt.show()












def groupedBarPlotStateLoop(savestring, root, firstseed, num_trials, bfi):
    

    # savestring = 'healthcare_10_1'
    # n_list = [100, 250, 500, 750]
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]

    # savestring = 'eng11_trials_resubmit'
    # n_list = [100, 200, 500, 750]
    # percent_lam0_list = [0.5, 0.8]
    # state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]


    policies_to_plot = [0, 24, 27, 21, 38, 39, 40]
    

    plot_data = np.zeros((len(state_size_list), len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(state_size_list), len(policies_to_plot), len(n_list)))


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

    elif x_axis_label == 'Budget (as fraction of N)':
        problem_size = n_list[3]
        x_axis = budget_frac_list
        state_size = state_size_list[1]

    elif x_axis_label == 'Number of states':
        problem_size = n_list[3]
        budget_frac - budget_frac_list[1]
        x_axis = state_size_list



    for k,state_size in enumerate(state_size_list):
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
                plot_data[k,i,x] = rewards.mean()
                errors[k,i,x] = rewards.std()


    labels = x_axis


    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.12  # the width of the bars
    
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
    fig, ax = plt.subplots(1,3, figsize=(14,3.2))
    for k,state_size in enumerate(state_size_list):
        for i,p in enumerate(policies_to_plot):
            rects1 = ax[k].bar(x+i*width, plot_data[k,i], width, yerr=errors[k,i], label=pname[p], 
                color=colors[i%4], edgecolor='k', linewidth='0.5', hatch=patterns[i//4], capsize=4)
            rects.append(rects1)

    for k in range(len(state_size_list)):
        ymin = plot_data[k][np.logical_not(np.isnan(plot_data[k]))].min() - 0.4
        ymax = plot_data[k][np.logical_not(np.isnan(plot_data[k]))].max() + 0.2

        ax[k].set_ylim([ymin,ymax])

    ylabel = 'Disc. sum rewards'
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')
    # ax.set_title(title, fontsize=14)   
    for k in range(len(state_size_list)):
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(labels, rotation=30)

    lsize=20
    fig.text(0.5, 0.03, 'Number of Arms (N)', ha='center')

    ax[1].legend(ncol=7, bbox_to_anchor =(0.5, 1.17),loc='center')
       
    plt.tight_layout() 
    plt.subplots_adjust(
                    wspace=0.1, 
                    hspace=0.0)

    # plt.title('B:%s'%budget_frac)


    outname = '%s_b%.1f_stateloop_rewards.pdf'%(savestring,budget_frac)
    plt.savefig(outname)
    plt.show()





def groupedBarPlotBudgetLoop(savestring, root, firstseed, num_trials, ssi):
    

    # savestring = 'healthcare_10_1'
    # n_list = [100, 250, 500, 750]
    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]

    # savestring = 'eng11_trials_resubmit'
    # n_list = [100, 200, 500, 750]
    # percent_lam0_list = [0.5, 0.8]
    # state_size_list = [11, 21, 31] # actually the n_actions_list = [10, 20, 30]


    policies_to_plot = [0, 24, 27, 21, 38, 39, 40]
   

    plot_data = np.zeros((len(budget_frac_list), len(policies_to_plot), len(n_list)))
    errors = np.zeros((len(budget_frac_list), len(policies_to_plot), len(n_list)))


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
        state_size = state_size_list[ssi]

    elif x_axis_label == 'Budget (as fraction of N)':
        problem_size = n_list[3]
        x_axis = budget_frac_list
        state_size = state_size_list[1]

    elif x_axis_label == 'Number of states':
        problem_size = n_list[3]
        budget_frac - budget_frac_list[1]
        x_axis = state_size_list



    for k,budget_frac in enumerate(budget_frac_list):
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
                plot_data[k,i,x] = rewards.mean()
                errors[k,i,x] = rewards.std()


    labels = x_axis


    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.12  # the width of the bars
    
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
    fig, ax = plt.subplots(1,3, figsize=(14,3.2))
    for k,budget_frac in enumerate(budget_frac_list):
        for i,p in enumerate(policies_to_plot):
            rects1 = ax[k].bar(x+i*width, plot_data[k,i], width, yerr=errors[k,i], label=pname[p], 
                color=colors[i%4], edgecolor='k', linewidth='0.5', hatch=patterns[i//4], capsize=4)
            rects.append(rects1)

    for k in range(len(budget_frac_list)):
        ymin = plot_data[k][np.logical_not(np.isnan(plot_data[k]))].min() - 0.4
        ymax = plot_data[k][np.logical_not(np.isnan(plot_data[k]))].max() + 0.2

        ax[k].set_ylim([ymin,ymax])

    ylabel = 'Disc. sum rewards'
    ax[0].set_ylabel(ylabel)
    ax[0].set_xlabel('(a)')
    ax[1].set_xlabel('(b)')
    ax[2].set_xlabel('(c)')

    for k in range(len(budget_frac_list)):
        ax[k].set_xticks(x)
        ax[k].set_xticklabels(labels, rotation=30)

    lsize=20
    fig.text(0.5, 0.03, 'Number of Arms (N)', ha='center')

    ax[1].legend(ncol=7, bbox_to_anchor =(0.5, 1.17),loc='center')
       
    plt.tight_layout() 
    plt.subplots_adjust(
                    wspace=0.1, 
                    hspace=0.0)
    # plt.title('S:%s'%state_size)

    outname = '%s_b%.1f_budgetloop_rewards.pdf'%(savestring,budget_frac)
    plt.savefig(outname)
    plt.show()






# savestring = 'healthcare_10_1'
savestring = 'healthcare_2_19'

root='./../batches/%s'%savestring
firstseed=0
# num_trials=25
num_trials = 20


for i in range(3):
    runtimePlotLoopStates(savestring, root, firstseed, num_trials, i, ignore_missing=True)
    groupedBarPlotStateLoop(savestring, root, firstseed, num_trials, i)
    runtimePlotLoopBudgets(savestring, root, firstseed, num_trials, i, ignore_missing=True)
    groupedBarPlotBudgetLoop(savestring, root, firstseed, num_trials, i)


