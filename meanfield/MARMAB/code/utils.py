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
# mpl.rcParams['ps.useafm'] = True
# mpl.rcParams['pdf.use14corefonts'] = True
# mpl.rcParams['text.usetex'] = True
# mpl.rcParams['text.latex.unicode']=True


# SMALL_SIZE = 30
# MEDIUM_SIZE = 36
# BIGGER_SIZE = 36
# plt.rc('font', weight='bold')
# plt.rc('font', size=BIGGER_SIZE)          # controls default text sizes
# plt.rc('axes', titlesize=BIGGER_SIZE)    # fontsize of the axes title
# plt.rc('axes', labelsize=BIGGER_SIZE)    # fontsize of the x and y labels
# plt.rc('xtick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('ytick', labelsize=BIGGER_SIZE)    # fontsize of the tick labels
# plt.rc('legend', fontsize=SMALL_SIZE)    # legend fontsize
# plt.rc('figure', titlesize=BIGGER_SIZE)  # fontsize of the figure title


@jit(nopython=True)
def list_valid_action_combinations(N,C,B,options):

    costs = np.zeros(options.shape[0])
    for i in range(options.shape[0]):
        costs[i] = C[options[i]].sum()
    valid_options = costs <= B
    options = options[valid_options]
    return options


def epsilon_clip(T, epsilon):
    return np.clip(T, epsilon, 1-epsilon)


    
def groupedBarPlot(infile_prefix, ylabel='Average Adherence out of 180 days',
            title='', filename='image.png', root='.'):
    
    import glob
    d={}
    labels=[]
    for fname in glob.glob(infile_prefix+'*'):
        df = pd.read_csv(fname)
        d[fname] = {}
        d[fname]['labels'] = df.columns.values
        labels = df.columns.values
        d[fname]['values'] = df.values[0]
        d[fname]['errors'] = df.values[1]

    print(d)

    fname = os.path.join(root,'test.png')

    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.2  # the width of the bars
    
    rects = []
    fig, ax = plt.subplots(figsize=(8,6))
    for i,key in enumerate(d.keys()):
        rects1 = ax.bar(x+i*width, d[key]['values'], width, yerr=d[key]['errors'], label='average adherence'+key[-8:])
        rects.append(rects1)
    ax.set_ylabel(ylabel)
    ax.set_title(title)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=60)
    ax.legend()
    
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
    plt.savefig(fname)
    # plt.show()

def barPlot(labels, values, errors, ylabel='',
            title='', filename='image.png', root='.',
            bottom=0):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    x = np.arange(len(labels))  # the label locations
    width = 0.85  # the width of the bars
    fig, ax = plt.subplots(figsize=(8,5))
    rects1 = ax.bar(x, values, width, yerr=errors, bottom=bottom, label='average adherence')
    # rects1 = ax.bar(x, values, width, bottom=bottom, label='Intervention benefit')
    
    ax.set_ylabel(ylabel, fontsize=14)
    ax.set_title(title, fontsize=14)   
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=30)
    ax.legend()
    
    def autolabel(rects):
        """Attach a text label above each bar in *rects*, displaying its height."""
        for rect in rects:
            height = rect.get_height()
            ax.annotate('{}'.format(height),
                        xy=(rect.get_x() + rect.get_width() / 2, height),
                        xytext=(0, 3),  # 3 points vertical offset
                        textcoords="offset points",
                        ha='center', va='bottom')
            
    # autolabel(rects1)       
    # plt.tight_layout() 
    plt.gcf().subplots_adjust(bottom=0.15)
    plt.savefig(fname)
    plt.show()


def plotLambdas(true_values, decoupled_values, filename, root):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(true_values)):
        true_lam, decoupled_lams = true_values[i], np.array(decoupled_values[i])

        # print('True vs others')
        # print(true_lam)
        # print(decoupled_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        vals, bins, p = ax.hist(decoupled_lams, 50, label=r'$\lambda^{i}$')

        ax.plot([true_lam, true_lam],[0,max(vals)], label=r'$\lambda_{min}$', linestyle='--', linewidth=5)

        # mean_decoupled_lam = np.mean(decoupled_lams[decoupled_lams>0])
        # ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label='Nonzero Mean decoupled lambda', linestyle='-.')

        mean_decoupled_lam = np.mean(decoupled_lams)
        ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label=r'Mean($\lambda^{i}$)', linestyle='-.', linewidth=5)




        ax.set_ylabel('Count')
        ax.set_xlabel(r'$\lambda^i$')
        # ax.set_title('Distribution of decoupled lambdas about the coupled value', fontsize=14)   
        ax.legend(loc='upper right')
        
         
        plt.tight_layout() 
        plt.savefig(fname)
        plt.show()
        break


def plotIterativeLambdas(true_values, iterative_values, filename, root, only_goods_lambda=None):
    
    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(true_values)):
        true_lam, iter_lams = true_values[i], iterative_values[i]
        only_goods_lambda_val = None
        if only_goods_lambda is not None:
            only_goods_lambda_val = only_goods_lambda[i]

        # print('True vs others')
        # print(true_lam)
        # print(iter_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        
        ax.plot(iter_lams, label='iterative lambdas')
        ax.plot([0,len(iter_lams)], [true_lam, true_lam], label='Coupled lambda', linestyle='--')

        if only_goods_lambda is not None:
            ax.plot([0,len(iter_lams)], [only_goods_lambda_val, only_goods_lambda_val], label='Only "goods" lambda', linestyle=':')            
        # mean_decoupled_lam = np.mean(decoupled_lams)
        # ax.plot([mean_decoupled_lam, mean_decoupled_lam],[0,max(vals)], label='Mean decoupled lambda', linestyle='-.')

        ax.set_ylabel('lambda value', fontsize=14)
        ax.set_title('Progression of iterative lambdas against true coupled value', fontsize=14)   
        ax.legend()
        
         
        plt.tight_layout() 
        plt.savefig(fname%i)
        plt.show()


def plotBLambdas(lbs, ubs, xvals, true_lambdas, filename, root):

    fname = os.path.join(root,filename)
    # plt.figure(figsize=(8,6))
    for i in range(len(lbs)):
        lb, ub, xval, true_lam = lbs[i], ubs[i], xvals[i], true_lambdas[i]

        # print('True vs others')
        # print(true_lam)
        # print(iter_lams)
        fig, ax = plt.subplots(figsize=(8,5))
        
        ax.plot(xval, lb, label='Lower bound')
        ax.plot(xval, ub, label='Upper bound')
        ax.plot([xval[0],xval[-1]], [true_lam, true_lam], label='True lambda')

        ax.set_ylabel('lambda value', fontsize=14)
        ax.set_title('Progression of BLam', fontsize=14)   
        ax.legend()

        ax.set_ylim([0,0.5])
        
         
        plt.tight_layout() 
        plt.savefig(fname%i)
        plt.show()




