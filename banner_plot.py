"""For analyzing and plotting RMAB simulation results.

Author: killian-34/jakillian
"""

import argparse
import os
import pickle

import environments
from environments import DiabetesAppEnvironment
from environments import DiabetesAppPartialObsEnvironment
from environments import TwoStateEnvironment
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from policies import EquitableLPPolicyMMR
from policies import EquitableLPPolicyMNW
from policies import EquitableLPPolicyMNWEqualGroups
from policies import HighA1cRRPolicy
from policies import NoActionPolicy
from policies import RandomHighA1cPolicy
from policies import RandomPolicy
from policies import WhittleIndexPolicy
import util

parser = argparse.ArgumentParser(
    description='Plot data from experiments. Must first run combine_results.py'
)
parser.add_argument('--expname', type=str, help='Name of experiment')
parser.add_argument(
    '--model', type=str, help='Name of model data file for experiments'
)
parser.add_argument(
    '--params', type=str, help='Name of parameter file for experiments'
)
parser.add_argument(
    '--image_type',
    type=str,
    default='png',
    choices=['png', 'pdf', 'jpg'],
    help='Plot image file type',
)
parser.add_argument(
    '--csv',
    action='store_true',
    help='If true, will write csv summaries of data in plots',
)
args = parser.parse_args()

param_dict = util.parse_parameters(args.params)

IMAGE_TYPE = args.image_type

policy_name_conversion = {
    NoActionPolicy.policy_key(): 'No Act',
    RandomHighA1cPolicy.policy_key(): 'HR-Rand',
    HighA1cRRPolicy.policy_key(): 'HR-RR',
    RandomPolicy.policy_key(): 'Rand',
    WhittleIndexPolicy.policy_key(): 'Opt',
    EquitableLPPolicyMMR.policy_key(): 'MMR',
    EquitableLPPolicyMNW.policy_key(): 'MNW',
    EquitableLPPolicyMNWEqualGroups.policy_key(): 'MNW-EG',
}

policy_name_list_reverse = {
    val: key for key,val in policy_name_conversion.items()
}

#  "#00BFC4"  "#C77CFF"
policy_colors = {
    NoActionPolicy.policy_key(): '#000000',
    RandomPolicy.policy_key(): 'y',
    RandomHighA1cPolicy.policy_key(): '#F8766D',
    HighA1cRRPolicy.policy_key(): 'c',
    WhittleIndexPolicy.policy_key(): '#7CAE00',
    EquitableLPPolicyMMR.policy_key(): 'b',
    EquitableLPPolicyMNW.policy_key(): 'm',
    EquitableLPPolicyMNWEqualGroups.policy_key(): '#c07b3b',
}


def banner_offline(
    seed,
    n_trials,
    img_dir,
    expname,
    policy_name_list=None,
):
  """Group reward difference bar chart."""

  small_size = 12
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=medium_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

#   plt.rcParams['figure.figsize'] = (8, 3)
  

  # policy_name_list = [
  #     EquitableLPPolicyMMR.policy_key(),
  #     EquitableLPPolicyMMR.policy_key(),
  # ]
  policy_name_list = ['No Act','Rand','Opt','MNW','MMR','MNW-EG']
  n_policies = len(policy_name_list)

  #   barwidth = 0.14
  barwidth = 0.14

  # add one more spot for the arm means
  pol_mean_positions = np.arange(n_policies)
  pol_mean_positions = pol_mean_positions - pol_mean_positions.mean()
  pol_mean_positions = pol_mean_positions*barwidth
  bar_spacing = 1
  more_space = 1*bar_spacing

  # build the banner
  # row 1: counter, for 3 different budgets
  # row 2: maternal, for each of 3 data
  # row 3: dbapp, for 3 different alphas
  fig, axs = plt.subplots(3, 3, figsize=(16, 5))

  # row 1:
  budget_fracs = [0.1, 0.2, 0.33]
  row = 0
  n_arms = 100

  ymax_gini = 0
  ymax_reward = 0

  ax_pairs = []

  for b, budget_frac in enumerate(budget_fracs):
    budget = int(n_arms * budget_frac)
    mad_score_file = 'outputs/csv_summaries/counterexample/group_mad_scores_seed-0_ntrials-%i_narms100_budget%i.csv'%(n_trials, budget)
    reward_file = 'outputs/csv_summaries/counterexample/health_bar_chart_seed-0_ntrials-%i_narms100_b%.2f_means.csv'%(n_trials, budget)
    reward_file_errors = 'outputs/csv_summaries/counterexample/health_bar_chart_seed-0_ntrials-%i_narms100_b%.2f_errors.csv'%(n_trials, budget)
    mads = pd.read_csv(mad_score_file)
    rs = pd.read_csv(reward_file, index_col=0)
    rse = pd.read_csv(reward_file_errors, index_col=0)
    axs2 = axs[row, b].twinx()
    ax_pairs.append(axs2)
    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      mad = mads[policy_name].values[0]
      axs[row, b].bar(
          pol_mean_positions[i] + 1 * bar_spacing,
          mad,
          # yerr=rewards_errors[i, :-1],
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          color=policy_colors[pname],
          label=policy_name  # + ', Gini=%.3f' % mad,
      )

    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      reward = rs.loc[policy_name]['Value']/n_arms
      reward_e = rse.loc[policy_name]['Value']/n_arms
      axs2.bar(
          pol_mean_positions[i] + 1 * bar_spacing + more_space,
          reward,
          yerr=reward_e,
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          hatch='/',
          color=policy_colors[pname],
          #   label=policy_name_list_pretty[i] + ' %.3f' % mad,
      )

    ymax_gini = max(ymax_gini, mads.values.max())
    ymax_reward = max(ymax_reward, rs.values.max()/n_arms)


  def rnd(a):
    return round(a, 2)

  # fig.yaxis.set_label_coords(-.5, 0)
  for b in range(len(budget_fracs)):
    axs[row, b].set_ylim([0, ymax_gini*1.1])
    ax_pairs[b].set_ylim([0, ymax_reward*1.1])

    n_ticks = 4
    upper = ymax_gini*1.1
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(rnd, ticks))
    axs[row, b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      axs[row, b].set_yticks(ticks, [])

    upper = ymax_reward*1.1
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(int, ticks))
    ax_pairs[b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      ax_pairs[b].set_yticks(ticks, [])  # Set label locations.

    x_pos = 1
    y_pos = ymax_gini*1.2
    axs[row, b].text(x_pos, y_pos, 'Gini', horizontalalignment='center')

    x_pos =  2
    # y_pos = rewards_means[:,-1].max()*1.5
    axs[row, b].text(x_pos, y_pos, 'Reward', horizontalalignment='center')

    axs[row, b].set_xticks([])

    x = 0.5+1 * bar_spacing
    maxval = ymax_gini
    axs[row, b].plot([x,x],[0,maxval],linestyle='--')
    axs[row, b].grid(axis='y')
  

  # row 2:
  big_groups = ['A', 'B', 'C']
  row = 1
  n_arms = 200
  budget = 20

  ymax_gini = 0
  ymax_reward = 0

  ax_pairs = []

  for b, big_group in enumerate(big_groups):
    # budget = int(n_arms * budget_frac)
    mad_score_file = 'outputs/csv_summaries/maternal_%s/group_mad_scores_seed-0_ntrials-%i_narms%s_budget%i.csv'%(big_group, n_trials, n_arms, budget)
    reward_file = 'outputs/csv_summaries/maternal_%s/health_bar_chart_seed-0_ntrials-%i_narms%s_b%.2f_means.csv'%(big_group, n_trials, n_arms, budget)
    reward_file_errors = 'outputs/csv_summaries/maternal_%s/health_bar_chart_seed-0_ntrials-%i_narms%s_b%.2f_errors.csv'%(big_group, n_trials, n_arms, budget)
    mads = pd.read_csv(mad_score_file)
    rs = pd.read_csv(reward_file, index_col=0)
    rse = pd.read_csv(reward_file_errors, index_col=0)
    axs2 = axs[row, b].twinx()
    ax_pairs.append(axs2)
    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      mad = mads[policy_name].values[0]
      axs[row, b].bar(
          pol_mean_positions[i] + 1 * bar_spacing,
          mad,
          # yerr=rewards_errors[i, :-1],
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          color=policy_colors[pname],
          label=policy_name  # + ', Gini=%.3f' % mad,
      )

    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      reward = rs.loc[policy_name]['Value']/n_arms
      reward_e = rse.loc[policy_name]['Value']/n_arms
      axs2.bar(
          pol_mean_positions[i] + 1 * bar_spacing + more_space,
          reward,
          yerr=reward_e,
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          hatch='/',
          color=policy_colors[pname],
          #   label=policy_name_list_pretty[i] + ' %.3f' % mad,
      )

    ymax_gini = max(ymax_gini, mads.values.max())
    ymax_reward = max(ymax_reward, rs.values.max()/n_arms)



  def rnd(a):
    return round(a, 2)

  def rndr(a):
    return round(a, 1)

  # fig.yaxis.set_label_coords(-.5, 0)
  for b in range(len(budget_fracs)):
    axs[row, b].set_ylim([0, ymax_gini*1.1])
    ax_pairs[b].set_ylim([0, ymax_reward*1.1])

    n_ticks = 4
    upper = ymax_gini*1.1
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(rnd, ticks))
    axs[row, b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      axs[row, b].set_yticks(ticks, [])

    upper = ymax_reward*1.111
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(rndr, ticks))
    ax_pairs[b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      ax_pairs[b].set_yticks(ticks, [])  # Set label locations.


    axs[row, b].set_xticks([])

    x = 0.5+1 * bar_spacing
    maxval = ymax_gini
    axs[row, b].plot([x,x],[0,maxval],linestyle='--')
    axs[row, b].grid(axis='y')

  
  # row 3:
  policy_name_list = ['No Act','Rand','HR-RR','Opt','MNW','MMR','MNW-EG']
  barwidth = 0.1

  n_policies = len(policy_name_list)

  # add one more spot for the arm means
  pol_mean_positions = np.arange(n_policies)
  pol_mean_positions = pol_mean_positions - pol_mean_positions.mean()
  pol_mean_positions = pol_mean_positions*barwidth
  bar_spacing = 1
  more_space = 1*bar_spacing

  alphas = [0, 0.5, 1.0]
  row = 2
  n_arms = 300
  budget = 30

  ymax_gini = 0
  ymax_reward = 0

  ax_pairs = []

  for b, alpha in enumerate(alphas):
    # budget = int(n_arms * budget_frac)
    mad_score_file = 'outputs/csv_summaries/DBApp/group_mad_scores_seed-0_ntrials-%i_narms%s_budget%i_alpha%.2f.csv'%(n_trials, n_arms, budget, alpha)
    reward_file = 'outputs/csv_summaries/DBApp/health_bar_chart_seed-0_ntrials-%i_narms%s_b%.2f_alpha%.2f_means.csv'%(n_trials, n_arms, budget, alpha)
    reward_file_errors = 'outputs/csv_summaries/DBApp/health_bar_chart_seed-0_ntrials-%i_narms%s_b%.2f_alpha%.2f_errors.csv'%(n_trials, n_arms, budget, alpha)
    mads = pd.read_csv(mad_score_file)
    rs = pd.read_csv(reward_file, index_col=0)
    rse = pd.read_csv(reward_file_errors, index_col=0)
    axs2 = axs[row, b].twinx()
    ax_pairs.append(axs2)
    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      mad = mads[policy_name].values[0]
      axs[row, b].bar(
          pol_mean_positions[i] + 1 * bar_spacing,
          mad,
          # yerr=rewards_errors[i, :-1],
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          color=policy_colors[pname],
          label=policy_name  # + ', Gini=%.3f' % mad,
      )

    for i, policy_name in enumerate(policy_name_list):
      pname = policy_name_list_reverse[policy_name]
      reward = rs.loc[policy_name]['Value']/n_arms
      reward_e = rse.loc[policy_name]['Value']/n_arms
      axs2.bar(
          pol_mean_positions[i] + 1 * bar_spacing + more_space,
          reward,
          yerr=reward_e,
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          hatch='/',
          color=policy_colors[pname],
          #   label=policy_name_list_pretty[i] + ' %.3f' % mad,
      )

    ymax_gini = max(ymax_gini, mads.values.max())
    ymax_reward = max(ymax_reward, rs.values.max()/n_arms)



  def rnd(a):
    return round(a, 3)

  # fig.yaxis.set_label_coords(-.5, 0)
  for b in range(len(budget_fracs)):
    axs[row, b].set_ylim([0, ymax_gini*1.1])
    ax_pairs[b].set_ylim([0, ymax_reward*1.1])

    n_ticks = 4
    upper = ymax_gini*1.1
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(rnd, ticks))
    axs[row, b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      axs[row, b].set_yticks(ticks, [])

    upper = ymax_reward*1.11111
    ticks = np.arange(0, upper, step=upper/n_ticks)
    ticks = list(map(int, ticks))
    ax_pairs[b].set_yticks(ticks)  # Set label locations.
    if b > 0:
      ax_pairs[b].set_yticks(ticks, [])  # Set label locations.


    axs[row, b].set_xticks([])

    x = 0.5+1 * bar_spacing
    maxval = ymax_gini
    axs[row, b].plot([x,x],[0,maxval],linestyle='--')
    axs[row, b].grid(axis='y')



  # fig.supylabel('Mean %s Reward' % REWARDS_AGGREGATION_METHOD[1])
  # axs[0].set_ylabel('Mean %s Reward' % REWARDS_AGGREGATION_METHOD[1])
  # axs[0].yaxis.set_label_coords(-.1, -.1)


  # time_delta = horizon
  # if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
  #   time_delta = REWARDS_AGGREGATION_METHOD[2]
  
  axs[2, 1].legend(ncol=n_policies, loc='lower center', bbox_to_anchor=(0.5, -0.59))
  # fig.subplots_adjust(top=0.75, left=0.2, bottom=0.1, right=0.95)

  sub_dir_name = 'banner_plots'
  sub_dir_path = os.path.join(img_dir, sub_dir_name)
  if not os.path.exists(sub_dir_path):
    os.makedirs(sub_dir_path)

  
  save_str = '%s/banner_seed-%s_ntrials-%s.%s'
  plt.savefig(
      save_str
      % (
          sub_dir_path,
          seed,
          n_trials,
          IMAGE_TYPE,
      ),
      dpi=200,
  )
  plt.clf()
  # fig = plt.gcf()
  plt.close(fig)



if __name__ == '__main__':

  batch_dir_path_in = util.Config.get_img_output_path(args.expname)

  n_trials_in = param_dict['n_trials']
  base_seed_in = param_dict['base_seed']
  ########
  # High-level analyses of final outcomes and capacity planning
  ########
  banner_offline(
    base_seed_in,
    n_trials_in,
    batch_dir_path_in,
    expname=args.expname,
    policy_name_list=param_dict['policy_list'],
  )

#   else:
#     err_str = 'Analysis not implemented for env "%s"' % environment_name_in
#     raise ValueError(err_str)
