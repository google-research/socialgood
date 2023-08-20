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

csv = args.csv

# Reward aggregation method, defined by a 3-tuple as follows:
#  0: {'From Start', 'Timepoints'}
#     'From Start' counts from the beginning of the simulation
#     'Timepoints' counts from the point each arm became active in simulation
#  1: {'Final', 'Total'}
#     'Final' reports the final state/reward of each arm
#     'Total' reports the sum of rewards of each arm over the accounting period
#  2: int defining the length of the accounting period

horizon = param_dict['horizon_list'][0]
REWARDS_AGGREGATION_METHOD = ('From Start', 'Total', horizon)
# REWARDS_AGGREGATION_METHOD = ('From Start', 'Final', horizon)
# REWARDS_AGGREGATION_METHOD = ('From Start', 'Final', 12)
# REWARDS_AGGREGATION_METHOD = ('Timepoints', 'Total', 12)
# REWARDS_AGGREGATION_METHOD = ('Timepoints', 'Final', 6)
# REWARDS_AGGREGATION_METHOD = ('Timepoints', 'Final', 12)


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


def reward_accounting(
    reward_list, state_list, accounting_method, ind_to_tup_dict
):
  """Implement reward accounting methods.

  Args:
    reward_list: array of rewards of dimension n_trials x n_arms x horizon
    state_list: array of states of dimension n_trials x n_arms x horizon
    accounting_method: a 3-tuple, as follows:
      0: {'From Start', 'Timepoints'}
        'From Start' counts from the beginning of the simulation
        'Timepoints' counts from the point each arm became active in simulation
      1: {'Final', 'Total'}
        'Final' reports the final state/reward of the accoutning period
        'Total' reports the sum of rewards over the accounting period
      2: int that defines the length of the accounting period
    ind_to_tup_dict: dictionary mapping state indexes to tuples

  Returns:
    rewards aggregated according to accounting_method
  """
  rewards_accounted = None
  if accounting_method[:2] == ('From Start', 'Final'):
    rewards_accounted = reward_list[:, :, -1]  # final step rewards
  elif accounting_method[:2] == ('From Start', 'Total'):
    rewards_accounted = reward_list.sum(axis=-1)  # sum over horizon rewards
  elif accounting_method[0] == 'Timepoints':
    rewards_accounted = returns_after_time_delta_months(
        state_list,
        reward_list,
        accounting_method[1],
        accounting_method[2],
        ind_to_tup_dict,
    )
  else:
    print('Accounting method not recognized')
    exit()

  return rewards_accounted


def returns_after_time_delta_months(
    state_list, reward_list, aggregation_method, time_delta, ind_to_tup_dict
):
  """Return rewards over a period of x months, starting from first active point.
  """
  n_trials = state_list.shape[0]
  n_arms = state_list.shape[1]
  horizon = state_list.shape[2]

  returns_out = np.zeros((n_trials, n_arms))

  eng_do_sc_trajectories = state_ind_eng_do_sc_combined(
      state_list, ind_to_tup_dict
  )

  inactive_trajectories = np.copy(eng_do_sc_trajectories)
  inactive_trajectories[inactive_trajectories != 3] = 0

  # * start state is the last day you have entry 3 in the eng_do_sc trajectories
  # * since 3 is the largest entry we deal with, solve this with reverse argmax
  #     search
  # * finally, add a dummy value of 3 to the front, since we want to shift
  #     indexes by 1 and so that arms that started on day 0 are properly handled
  dummy = 3 * np.ones((n_trials, n_arms, 1))
  inactive_trajectories = np.concatenate([dummy, inactive_trajectories], axis=2)
  start_times = horizon - np.argmax(inactive_trajectories[:, :, ::-1], axis=2)

  # loop for now, maybe there is a better way
  # Note: if arm was active for less time than time_delta, this method
  #   will account all their active data, or return their state from their final
  #   day
  for t in range(n_trials):
    for n in range(n_arms):
      s = start_times[t, n]
      reward_subsequence = reward_list[t, n, s : s + time_delta]
      if aggregation_method == 'Final':
        returns_out[t, n] = reward_subsequence[-1]
      elif aggregation_method == 'Total':
        returns_out[t, n] = reward_subsequence.sum()

  return returns_out


def state_to_reward(states, rewards, ind):
  rewards_out = np.zeros(states.shape)
  for t in range(states.shape[0]):
    for arm in range(states.shape[1]):
      for h in range(states.shape[2]):
        state_ind = states[t, arm, h]
        rewards_out[t, arm, h] = rewards[arm, state_ind, ind]
  return rewards_out


def state_ind_to_tup_eng_health(
    states, state_index_to_tuple_dict, include_memory_states=False
):
  """Convert state indexes to a tuple of engagement and health states."""
  env_class = DiabetesAppPartialObsEnvironment
  new_dropout_def = 2
  state_tups = np.zeros(states.shape, dtype=object)

  for t in range(states.shape[0]):
    for arm in range(states.shape[1]):
      for h in range(states.shape[2]):
        state_tup = state_index_to_tuple_dict[states[t, arm, h]]

        s_e = state_tup[0]
        s_m = state_tup[1]
        s_t = state_tup[2]
        s_d = state_tup[3]
        s_h = state_tup[4]
        s_c = state_tup[5]

        eng_state = s_e
        if s_t == 0:
          eng_state = s_e
          # can't be in dropout at head of chain
        elif s_t >= 1:
          if s_d == env_class.DROPOUT:
            eng_state = new_dropout_def  # combine dropout and eng states...
          elif s_d == env_class.NODROPOUT:
            eng_state = env_class.MAINTENANCE  # if s_t>1, not eng

        # in partially observable world, dynamics driven by
        # chains and counters
        if s_t == 0:
          mem_tup = s_m
        elif s_t == 1:
          mem_tup = (s_e, s_m[0])  # advance memory by 1
        elif s_t == 2:
          mem_tup = (env_class.MAINTENANCE, s_e)  # advance memory by 2

        if include_memory_states:
          state_tup_eng_health = (
              eng_state,
              s_h,
              mem_tup,
              s_c,
          )  # only keep eng, health, and memory
        else:
          state_tup_eng_health = (
              eng_state,
              s_h,
              s_c,
          )  # only keep eng and health

        state_tups[t, arm, h] = state_tup_eng_health

  return state_tups


def state_ind_eng_do_sc_combined(states, state_index_to_tuple_dict):
  """Convert indexes to encoding for engagement, dropout, and stream states."""
  env_class = DiabetesAppPartialObsEnvironment
  new_dropout_def = 2
  new_inactive_def = 3
  eng_do_sc_codes = np.zeros(states.shape, dtype=int)

  for t in range(states.shape[0]):
    for arm in range(states.shape[1]):
      for h in range(states.shape[2]):
        state_tup = state_index_to_tuple_dict[states[t, arm, h]]

        s_e = state_tup[0]
        # s_m = state_tup[1]
        s_t = state_tup[2]
        s_d = state_tup[3]
        # s_h = state_tup[4]
        s_c = state_tup[5]

        eng_state = s_e
        if s_t == 0:
          eng_state = s_e
          # can't be in dropout at head of chain
        elif s_t >= 1:
          if s_d == env_class.DROPOUT:
            eng_state = new_dropout_def  # combine dropout and eng states...
          elif s_d == env_class.NODROPOUT:
            eng_state = env_class.MAINTENANCE  # if s_t>1, not eng

        if s_c > 0:
          eng_state = new_inactive_def

        eng_do_sc_codes[t, arm, h] = eng_state

  return eng_do_sc_codes


def states_in_words(states, include_memory_states=False):
  """Convert engagement/health state tuples for strings."""
  eng_dict = {
      0: 'M',
      1: 'E',
      2: 'D',
  }
  health_dict = {0: '>8', 1: '<8'}
  tups_out = []
  for tup in states:
    s_e = eng_dict[tup[0]]
    s_h = health_dict[tup[1]]
    if include_memory_states:
      s_m = ''.join([eng_dict[s] for s in tup[2]])
      s_m = '[%s]' % s_m
      final_str = '%s %s\n%s' % (s_e, s_h, s_m)
    else:
      final_str = '%s %s' % (s_e, s_h)
    tups_out.append(final_str)
  return tups_out


def get_action_distros_by_alpha_with_state_distros(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    include_memory_states=False,
    expname=None,
):
  """Get action distributions by alpha with state distros."""
  colors_dict = {
      'state_distribution': 'r',
      'action_distribution': 'b',
  }

  for budget_frac in budget_frac_list:
    for policy_name in policy_name_list:
      budget = int(n_arms * budget_frac)
      # alphas_list = [0, 0.5, 1.0]
      n_alphas = len(alphas_list)

      if include_memory_states:
        _, axs = plt.subplots(n_alphas, 1, figsize=(16, 18))
      else:
        _, axs = plt.subplots(n_alphas, 1, figsize=(6, 9))

      if n_alphas < 2:
        axs = np.array([axs])

      for i, alpha in enumerate(alphas_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        data = problem_pickle['data']

        state_list = data[policy_name]['states']
        action_list = data[policy_name]['actions']
        state_list_tups = state_ind_to_tup_eng_health(
            state_list,
            state_index_to_tuple_dict=state_index_to_tuple_dict,
            include_memory_states=include_memory_states,
        )

        unique_tup_list = list(set(state_list_tups.reshape(-1)))

        # remove tuples for arms were not active (streaming)
        unique_tup_list = [tup for tup in unique_tup_list if tup[-1] == 0]
        sorted_tuples = sorted(unique_tup_list)

        tuple_action_counts = {tup: np.zeros(n_trials) for tup in sorted_tuples}
        tuple_counts = {tup: np.zeros(n_trials) for tup in sorted_tuples}
        # count all the appearances of tuples
        # and times each tuple received an action, per trial
        for t in range(n_trials):
          for arm in range(n_arms):
            for h in range(horizon):
              state_tup = state_list_tups[t, arm, h]
              if state_tup[-1] == 0:  # arm was active
                action = action_list[t, arm, h]
                tuple_counts[state_tup][t] += 1
                if action:
                  tuple_action_counts[state_tup][t] += 1

        cut_horizon = horizon
        tuple_count_stats = {
            tup: tuple_counts[tup].sum() / (n_trials * n_arms * cut_horizon)
            for tup in sorted_tuples
        }

        tuple_action_stats = {
            tup: tuple_action_counts[tup].sum() / (
                n_trials * n_arms * cut_horizon
            )
            for tup in sorted_tuples
        }

        xpos = np.arange(len(sorted_tuples))

        axs[i].bar(
            xpos,
            [tuple_count_stats[tup] for tup in sorted_tuples],
            # yerr=[tuple_counts[tup] for tup in sorted_tuples],
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color=colors_dict['state_distribution'],
        )
        axs[i].bar(
            xpos,
            [tuple_action_stats[tup] for tup in sorted_tuples],
            # yerr=[tuple_action_stats[tup]['error'] for tup in sorted_tuples],
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
            color=colors_dict['action_distribution'],
        )
        axs[i].set_xticks(xpos)
        axs[i].set_xticklabels(
            states_in_words(
                sorted_tuples, include_memory_states=include_memory_states
            )
        )
        axs[i].set_ylim([0, 0.5])  # 1.0])

        xpos = len(sorted_tuples) - 2
        axs[i].text(xpos, 0.4, r'$\alpha=%s$' % alpha)

      handle_tup_count = Line2D(
          [0],
          [0],
          label='State Distribution',
          color=colors_dict['state_distribution'],
      )
      handle_tup_action_count = Line2D(
          [0],
          [0],
          label='Effort Distribution',
          color=colors_dict['action_distribution'],
      )

      handles = [handle_tup_count, handle_tup_action_count]

      axs[0].legend(handles=handles)

      axs[n_alphas // 2].set_ylabel('Frequency')
      axs[0].set_title(
          'Effort Distribution (Sum over arms and time)\n n_arms=%s, budget=%s,'
          ' H=%s' % (n_arms, budget, horizon)
      )

      sub_dir_name = 'effort_distribution/state_action_overlay/%s'

      file_name_token = ''
      dir_token = 'engagement_health'
      if include_memory_states:
        file_name_token = '_with_memory'
        dir_token = 'engagement_health_memory'

      sub_dir_name = sub_dir_name % dir_token

      sub_dir_path = os.path.join(img_dir, sub_dir_name)
      if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

      save_str = '%s/effort_distribution_wstates_seed-%s_ntrials-'
      save_str += '%s_narms%s_p%s_budget%s%s.%s'
      plt.savefig(
          save_str
          % (
              sub_dir_path,
              seed,
              n_trials,
              n_arms,
              policy_name,
              budget,
              file_name_token,
              IMAGE_TYPE,
          )
      )
      plt.clf()
      fig = plt.gcf()
      plt.close(fig)


def get_action_distros_by_alpha_scaled_by_state_distros(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    include_memory_states=False,
    expname=None,
):
  """Get action distributions by alpha scaled by state distros."""

  for budget_frac in budget_frac_list:
    for policy_name in policy_name_list:
      budget = int(n_arms * budget_frac)
      # alphas_list = [0, 0.5, 1.0]
      n_alphas = len(alphas_list)

      if include_memory_states:
        _, axs = plt.subplots(n_alphas, 1, figsize=(16, 18))
      else:
        _, axs = plt.subplots(n_alphas, 1, figsize=(6, 9))

      if n_alphas < 2:
        axs = np.array([axs])

      for i, alpha in enumerate(alphas_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        data = problem_pickle['data']

        state_list = data[policy_name]['states']
        action_list = data[policy_name]['actions']
        state_list_tups = state_ind_to_tup_eng_health(
            state_list,
            state_index_to_tuple_dict=state_index_to_tuple_dict,
            include_memory_states=include_memory_states,
        )
        unique_tup_list = list(set(state_list_tups.reshape(-1)))
        # remove tuples for arms were not active (streaming)
        unique_tup_list = [tup for tup in unique_tup_list if tup[-1] == 0]
        sorted_tuples = sorted(unique_tup_list)

        tuple_action_counts = {
            tup: np.zeros((n_trials, n_arms, horizon)) for tup in sorted_tuples
        }
        tuple_counts = {
            tup: np.zeros((n_trials, n_arms, horizon)) for tup in sorted_tuples
        }
        scaled_tuple_action_counts = {
            tup: np.zeros((n_trials, n_arms, horizon)) for tup in sorted_tuples
        }

        # count all the appearances of tuples
        # and times each tuple received an action, per trial
        for t in range(n_trials):
          for h in range(horizon):
            for arm in range(n_arms):
              state_tup = state_list_tups[t, arm, h]
              if state_tup[-1] == 0:  # check arm was active (streaming)
                action = action_list[t, arm, h]
                tuple_counts[state_tup][t, arm, h] += 1
                if action:
                  tuple_action_counts[state_tup][t, arm, h] += 1

        #
        for tup in sorted_tuples:
          # do not divide by 0
          if tuple_counts[tup].sum() == 0:
            tuple_counts[tup][0, 0] = 1

          scaled_tuple_action_counts[tup] = (
              tuple_action_counts[tup].sum() / tuple_counts[tup].sum()
          )

        xpos = np.arange(len(sorted_tuples))

        # y_err = [
        #     scaled_tuple_action_counts[tup]['error'] for tup in sorted_tuples
        # ]
        axs[i].bar(
            xpos,
            [scaled_tuple_action_counts[tup] for tup in sorted_tuples],
            # yerr=y_err,
            align='center',
            alpha=0.5,
            ecolor='black',
            capsize=10,
        )

        axs[i].set_xticks(xpos)
        axs[i].set_xticklabels(
            states_in_words(sorted_tuples, include_memory_states)
        )
        axs[i].set_ylim([0, 1.0])
        xpos = len(sorted_tuples) - 2
        axs[i].text(xpos, 0.9, r'$\alpha=%s$' % alpha)

      axs[n_alphas // 2].set_ylabel('Effort Scaled by State Distribution')
      axs[0].set_title(
          'Effort Distribution (Sum over arms and time)\n n_arms=%s, budget=%s,'
          ' H=%s' % (n_arms, budget, horizon)
      )

      sub_dir_name = 'effort_distribution/scaled_by_states/%s'

      file_name_token = ''
      dir_token = 'engagement_health'
      if include_memory_states:
        file_name_token = '_with_memory'
        dir_token = 'engagement_health_memory'

      sub_dir_name = sub_dir_name % dir_token

      sub_dir_path = os.path.join(img_dir, sub_dir_name)
      if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

      save_str = '%s/effort_distribution_scaled_by_states_seed-'
      save_str += '%s_ntrials-%s_narms%s_p%s_budget%s%s.%s'
      plt.savefig(
          save_str
          % (
              sub_dir_path,
              seed,
              n_trials,
              n_arms,
              policy_name,
              budget,
              file_name_token,
              IMAGE_TYPE,
          )
      )
      plt.clf()
      fig = plt.gcf()
      plt.close(fig)


def get_state_distros_over_time(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Get engagement and health state trajectories over time."""

  for budget_frac in budget_frac_list:
    for policy_name in policy_name_list:
      budget = int(n_arms * budget_frac)
      # alphas_list = [0, 0.5, 1.0]
      n_alphas = len(alphas_list)

      # eng_trajectories = np.zeros((n_trials, horizon))
      health_trajectories = np.zeros((n_trials, horizon))

      _, axs = plt.subplots(n_alphas, 2, figsize=(6, 9))

      if n_alphas < 2:
        axs = axs.reshape(1, 2)

      for i, alpha in enumerate(alphas_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']
        state_index_to_tuple_dict = env_info['state_index_to_tuple']
        # state_tuple_to_index_dict = env_info['state_tuple_to_index']
        rewards_2d = env_info['rewards_2d']

        data = problem_pickle['data']

        # policy_name = WhittleIndexPolicySymmetricArms.policy_key()

        state_list = data[policy_name]['states']

        # 0 is M, 1 is E, 2 is DO, 3 is INACTIVE
        eng_do_sc_trajectories = state_ind_eng_do_sc_combined(
            state_list, state_index_to_tuple_dict
        )
        health_rewards = state_to_reward(state_list, rewards=rewards_2d, ind=1)

        # all remaining 1s represent states that were engaged
        engaged_trajectories = np.copy(eng_do_sc_trajectories)
        engaged_trajectories[eng_do_sc_trajectories != 1] = 0

        # all remaining 1s represent states that were in maintenance
        maintenance_trajectories = np.copy(eng_do_sc_trajectories)
        some_const = 10
        maintenance_trajectories[maintenance_trajectories != 0] = some_const
        maintenance_trajectories[maintenance_trajectories == 0] = 1
        maintenance_trajectories[maintenance_trajectories == some_const] = 0

        # all remaining 1s represent states that were in dropout
        dropout_trajectories = np.copy(eng_do_sc_trajectories)
        dropout_trajectories[dropout_trajectories != 2] = 0
        dropout_trajectories[dropout_trajectories == 2] = 1

        # all remaining 1s represent states that were in not active
        inactive_trajectories = np.copy(eng_do_sc_trajectories)
        inactive_trajectories[inactive_trajectories != 3] = 0
        inactive_trajectories[inactive_trajectories == 3] = 1

        # remaining 1s are states that were inactive or in dropout
        inactive_or_dropout_trajectories = np.copy(eng_do_sc_trajectories)
        inactive_or_dropout_trajectories[
            inactive_or_dropout_trajectories == 1
        ] = 0
        inactive_or_dropout_trajectories[
            inactive_or_dropout_trajectories == 2
        ] = 1  # count dropouts
        inactive_or_dropout_trajectories[
            inactive_or_dropout_trajectories == 3
        ] = 1  # count inactives

        # 0 := (A1c > 8), so flip
        # any arm inactive arm will also have 0 reward, so make
        # those 0 again after inverse
        health_trajectories = np.copy(health_rewards)
        health_trajectories = 1 - health_trajectories
        health_trajectories[inactive_trajectories == 1] = (
            0  # return inactives to 0
        )

        health_not_dropout_trajectories = np.copy(health_rewards)
        health_not_dropout_trajectories = 1 - health_not_dropout_trajectories
        health_not_dropout_trajectories[
            inactive_or_dropout_trajectories == 1
        ] = 0  # return inactives or dropouts to 0

        # compute normalizer over horizon, based on active arms
        num_inactive_arms = inactive_trajectories.sum(axis=1)  # sum over arms
        num_inactive_arms_over_horizon = num_inactive_arms.mean(
            axis=0
        )  # mean over trials should be same, just collapsing that dimension
        num_active_arms_over_horizon = n_arms - num_inactive_arms_over_horizon

        # compute normalizer over horizon based on active and
        # not dropped out arms
        num_inactive_and_dropout_arms = inactive_or_dropout_trajectories.sum(
            axis=1
        )  # sum over amrs
        num_inactive_and_dropout_arms_over_horizon = (
            n_arms - num_inactive_and_dropout_arms
        )  # keep the trials dimension here, since dropouts are random

        engaged_trajectories = (
            engaged_trajectories.sum(axis=1) / num_active_arms_over_horizon
        )  # average over arms
        maintenance_trajectories = (
            maintenance_trajectories.sum(axis=1) / num_active_arms_over_horizon
        )  # average over arms
        dropout_trajectories = (
            dropout_trajectories.sum(axis=1) / num_active_arms_over_horizon
        )  # average over arms
        health_trajectories = (
            health_trajectories.sum(axis=1) / num_active_arms_over_horizon
        )  # average over arms

        health_not_dropout_trajectories = (
            health_not_dropout_trajectories.sum(axis=1)
            / num_inactive_and_dropout_arms_over_horizon
        )  # average over arms

        # try:
        xpos = np.arange(horizon)

        colors_dict = {
            'engaged': 'g',
            'maintenance': 'm',
            'dropout': 'k',
            'health': 'r',
            'health_not_dropout': 'c',
        }

        axs[i, 0].plot(
            xpos,
            np.percentile(engaged_trajectories, q=50, axis=0),
            color=colors_dict['engaged'],
        )
        axs[i, 0].fill_between(
            xpos,
            np.percentile(engaged_trajectories, q=25, axis=0),
            np.percentile(engaged_trajectories, q=75, axis=0),
            color=colors_dict['engaged'],
            alpha=0.5,
        )
        axs[i, 0].plot(
            xpos,
            np.percentile(maintenance_trajectories, q=50, axis=0),
            color=colors_dict['maintenance'],
        )
        axs[i, 0].fill_between(
            xpos,
            np.percentile(maintenance_trajectories, q=25, axis=0),
            np.percentile(maintenance_trajectories, q=75, axis=0),
            color=colors_dict['maintenance'],
            alpha=0.5,
        )
        axs[i, 0].plot(
            xpos,
            np.percentile(dropout_trajectories, q=50, axis=0),
            color=colors_dict['dropout'],
        )
        axs[i, 0].fill_between(
            xpos,
            np.percentile(dropout_trajectories, q=25, axis=0),
            np.percentile(dropout_trajectories, q=75, axis=0),
            color=colors_dict['dropout'],
            alpha=0.5,
        )
        axs[i, 1].plot(
            xpos,
            np.percentile(health_trajectories, q=50, axis=0),
            color=colors_dict['health'],
        )
        axs[i, 1].fill_between(
            xpos,
            np.percentile(health_trajectories, q=25, axis=0),
            np.percentile(health_trajectories, q=75, axis=0),
            color=colors_dict['health'],
            alpha=0.5,
        )
        axs[i, 1].plot(
            xpos,
            np.percentile(health_not_dropout_trajectories, q=50, axis=0),
            color=colors_dict['health_not_dropout'],
        )
        axs[i, 1].fill_between(
            xpos,
            np.percentile(health_not_dropout_trajectories, q=25, axis=0),
            np.percentile(health_not_dropout_trajectories, q=75, axis=0),
            color=colors_dict['health_not_dropout'],
            alpha=0.5,
        )

        xticks = np.arange(horizon / 2) * 2
        axs[i, 0].set_xticks(xticks)
        axs[i, 1].set_xticks(xticks)
        # axs[i].set_xticklabels(states_in_words(sorted_tuples))
        # axs[i].set_xlabel(r"$\alpha=%s$"%alpha)
        axs[i, 0].set_ylim([0, 1.0])
        axs[i, 1].set_ylim([0, 1.0])

        alpha_pos = (horizon - 5, 0.9)
        axs[i, 0].text(alpha_pos[0], alpha_pos[1], r'$\alpha=%s$' % alpha)
        axs[i, 1].text(alpha_pos[0], alpha_pos[1], r'$\alpha=%s$' % alpha)
        # except Exception as e:
        #   print(e)
        #   import pdb

        #   pdb.set_trace()

      # plt.xlabel('State')
      # plt.ylabel('%% Effort per Simulation')
      axs[n_alphas // 2, 0].set_ylabel('Expected # arms in state')
      plt.suptitle(
          'State trajectories (Mean over arms, simulations)\n n_arms=%s,'
          ' budget=%s, H=%s' % (n_arms, budget, horizon)
      )

      line_eng = Line2D([0], [0], label='Engaged', color=colors_dict['engaged'])
      line_mnt = Line2D(
          [0], [0], label='Maintenance', color=colors_dict['maintenance']
      )
      line_do = Line2D([0], [0], label='Dropout', color=colors_dict['dropout'])
      line_health = Line2D(
          [0], [0], label='A1c > 8', color=colors_dict['health']
      )
      line_health_ndo = Line2D(
          [0],
          [0],
          label='A1c > 8 | (E or M)',
          color=colors_dict['health_not_dropout'],
      )

      handles = [line_eng, line_mnt, line_do, line_health, line_health_ndo]

      axs[0, 0].legend(
          handles=handles,
          bbox_to_anchor=(1.75, 1.4),
          ncol=2,
          fancybox=True,
          shadow=True,
      )

      sub_dir_name = 'state_trajectories'

      sub_dir_path = os.path.join(img_dir, sub_dir_name)
      if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

      # plt.tight_layout()
      plt.savefig(
          '%s/state_trajectories_seed-%s_ntrials-%s_narms%s_p%s_budget%s.%s'
          % (
              sub_dir_path,
              seed,
              n_trials,
              n_arms,
              policy_name,
              budget,
              IMAGE_TYPE,
          )
      )
      plt.clf()
      fig = plt.gcf()
      plt.close(fig)


def health_difference_bar_chart(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas,
    budget_fracs,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Health difference bar chart."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  rewards_means = np.zeros((n_policies))
  rewards_errors = np.zeros((n_policies))

  for budget_frac in budget_fracs:
    for alpha in alphas:
      budget = int(n_arms * budget_frac)
      for i, policy_name in enumerate(policy_name_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']

        data = problem_pickle['data']

        reward_list = data[policy_name][
            'rewards'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_list = data[policy_name][
            'states'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        reward_list_accounted = reward_accounting(
            reward_list,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )

        reward_arm_sum = reward_list_accounted.sum(axis=-1)  # sum over arms
        rewards_means[i] = reward_arm_sum.mean()
        rewards_errors[i] = reward_arm_sum.std() / np.sqrt(n_trials)

        plt.bar(
            i,
            rewards_means[i],
            yerr=rewards_errors[i],
            align='center',
            alpha=0.8,
            ecolor='black',
            capsize=10,
            color=policy_colors[policy_name],
            label=policy_name_list_pretty[i],
        )

      plt.xticks(np.arange(n_policies), policy_name_list_pretty, rotation=35)
      plt.ylabel('%s # Patients A1c<8' % (REWARDS_AGGREGATION_METHOD[1]))
      # plt.xlabel('Policy')
      time_delta = horizon
      if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
        time_delta = REWARDS_AGGREGATION_METHOD[2]

      plt.title(
          r'A1c After %s months, $\alpha$:%.2f' % (time_delta, alpha)
          + '\n'
          + '(%s Patients, %s Budget)' % (n_arms, budget)
      )
      plt.ylim([
          rewards_means.min() - 0.05 * n_arms,
          rewards_means.max() + 0.05 * n_arms,
      ])

      sub_dir_name = 'key_plots'
      sub_dir_path = os.path.join(img_dir, sub_dir_name)
      if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

      plt.subplots_adjust(top=0.8, left=0.2, bottom=0.3)

      # plt.tight_layout()
      save_str = '%s/health_bar_chart_seed-%s_ntrials-'
      save_str += '%s_narms%s_b%.2f_alpha%.2f_all_%s.%s'
      plt.savefig(
          save_str
          % (
              sub_dir_path,
              seed,
              n_trials,
              n_arms,
              budget,
              alpha,
              REWARDS_AGGREGATION_METHOD,
              IMAGE_TYPE,
          ),
          dpi=200,
      )
      plt.clf()
      fig = plt.gcf()
      plt.close(fig)

      if csv:
        sub_dir_path = util.Config.get_csv_summary_output_path(expname)

        fname = '%s/health_bar_chart_seed-%s_ntrials-%s_narms%s_b%.2f_alpha'
        fname += '%.2f_%s.csv'
        fname_means = fname % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
            'means',
        )
        fname_errors = fname % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
            'errors',
        )

        pretty_policy_name_list = [
            policy_name_conversion[policy_name]
            for policy_name in policy_name_list
        ]
        columns = ['Value']
        df_means = pd.DataFrame(
            rewards_means, index=pretty_policy_name_list, columns=columns
        )
        df_errors = pd.DataFrame(
            rewards_errors, index=pretty_policy_name_list, columns=columns
        )

        df_means.to_csv(fname_means)
        df_errors.to_csv(fname_errors)


def engagement_difference_bar_chart(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas,
    budget_fracs,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Engagement difference bar chart."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  rewards_means = np.zeros((n_policies))
  rewards_errors = np.zeros((n_policies))

  # alpha = 1.0
  alpha = alphas[-1]
  for budget_frac in budget_fracs:
    # for alpha in alphas:
    budget = int(n_arms * budget_frac)
    for i, policy_name in enumerate(policy_name_list):
      file_name = input_file_name_template % (
          expname,
          n_arms,
          budget_frac,
          environment_name,
          n_trials,
          seed,
          horizon,
          alpha,
      )

      problem_pickle = {}
      with open(file_name, 'rb') as fh:
        problem_pickle = pickle.load(fh)

      # environment_parameters = problem_pickle['simulation_parameters'][
      #     'environment_parameters'
      # ]

      env_info = problem_pickle['simulation_parameters']['environment_info']

      data = problem_pickle['data']

      reward_list = data[policy_name][
          'rewards'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
      state_list = data[policy_name][
          'states'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
      state_index_to_tuple_dict = env_info['state_index_to_tuple']

      reward_list_accounted = reward_accounting(
          reward_list,
          state_list,
          REWARDS_AGGREGATION_METHOD,
          state_index_to_tuple_dict,
      )

      not_dropped_out_arm_sum = reward_list_accounted.sum(
          axis=-1
      )  # sum over arms
      rewards_means[i] = not_dropped_out_arm_sum.mean()
      rewards_errors[i] = not_dropped_out_arm_sum.std() / np.sqrt(n_trials)

      plt.bar(
          i,
          rewards_means[i],
          yerr=rewards_errors[i],
          align='center',
          alpha=0.8,
          ecolor='black',
          capsize=10,
          color=policy_colors[policy_name],
          label=policy_name_list_pretty[i],
      )

    plt.xticks(np.arange(n_policies), policy_name_list_pretty, rotation=35)
    plt.ylabel('%s # Patients Active' % (REWARDS_AGGREGATION_METHOD[1]))
    time_delta = horizon
    if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
      time_delta = REWARDS_AGGREGATION_METHOD[2]

    plt.title(
        r'Engagement After %s months, $\alpha$:%.2f' % (time_delta, alpha)
        + '\n'
        + '(%s Patients, %s Budget)' % (n_arms, budget)
    )
    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.3)
    plt.ylim([
        rewards_means.min() - 0.05 * n_arms,
        rewards_means.max() + 0.05 * n_arms,
    ])

    sub_dir_name = 'key_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    # plt.tight_layout()
    save_str = '%s/engagement_bar_chart_seed-%s_ntrials-%s_narms%s_b%.2f'
    save_str += '_alpha%.2f_all_%s.%s'
    plt.savefig(
        save_str
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        ),
        dpi=200,
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/engagement_bar_chart_seed-%s_ntrials-%s_narms%s_b%.2f'
      fname += '_alpha%.2f_%s.csv'
      fname_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          alpha,
          'means',
      )
      fname_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          alpha,
          'errors',
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      columns = ['Value']
      df_means = pd.DataFrame(
          rewards_means, index=pretty_policy_name_list, columns=columns
      )
      df_errors = pd.DataFrame(
          rewards_errors, index=pretty_policy_name_list, columns=columns
      )

      df_means.to_csv(fname_means)
      df_errors.to_csv(fname_errors)


def group_reward_difference_bar_chart(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas,
    budget_fracs,
    img_dir,
    expname,
    policy_name_list=None,
    environment_class=None,
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
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  plt.rcParams['figure.figsize'] = (8, 6)

  # policy_name_list = [
  #     EquitableLPPolicyMMR.policy_key(),
  #     EquitableLPPolicyMMR.policy_key(),
  # ]
  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  _, group_mappings, group_definitions, _ = (
      environment_class.parse_model_data(args.model, n_arms)
  )
  group_list = sorted(list(set(group_mappings)))
  n_groups = len(group_list)

  rewards_means = np.zeros((n_policies, n_groups + 1))
  rewards_errors = np.zeros((n_policies, n_groups + 1))

  barwidth = 0.14

  group_positions = np.array([
      -2.5 * barwidth,
      -1.5 * barwidth,
      -0.5 * barwidth,
      0.5 * barwidth,
      1.5 * barwidth,
      2.5 * barwidth,
  ])
  bar_spacing = 1

  for budget_frac in budget_fracs:
    for alpha in alphas:
      budget = int(n_arms * budget_frac)

      mad_scores = np.zeros(n_policies)
      for i, policy_name in enumerate(policy_name_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']

        data = problem_pickle['data']

        reward_list = data[policy_name][
            'rewards'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_list = data[policy_name][
            'states'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        reward_list_accounted = reward_accounting(
            reward_list,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )

        health_end_of_horizon = reward_list_accounted

        for group in group_list:
          group_inds = group_mappings == group
          group_size = group_inds.sum()
          health_end_of_horizon_group = (
              health_end_of_horizon[:, group_inds].sum(axis=-1) / group_size
          )  # sum over arms in group
          rewards_means[i, group] = health_end_of_horizon_group.mean()
          rewards_errors[i, group] = (
              health_end_of_horizon_group.std() / np.sqrt(n_trials)
          )

        health_end_of_horizon = health_end_of_horizon.sum(
            axis=-1
        )  # sum over arms in g1
        rewards_means[i, n_groups] = health_end_of_horizon.mean() / n_arms
        rewards_errors[i, n_groups] = (
            health_end_of_horizon.std() / n_arms / np.sqrt(n_trials)
        )

        mad = 0
        for j in group_list:
          for k in group_list:
            mad += abs(rewards_means[i, j] - rewards_means[i, k])
        mad /= 2 * n_groups**2 * rewards_means[i].mean()

        plt.bar(
            group_positions + i * bar_spacing,
            rewards_means[i, :-1],
            yerr=rewards_errors[i, :-1],
            align='center',
            alpha=0.8,
            width=barwidth,
            edgecolor='black',
            ecolor='black',
            capsize=10,
            color=policy_colors[policy_name],
            label=policy_name_list_pretty[i] + ' %.3f' % mad,
        )
        mad_scores[i] = mad

      def nice_join(tup):
        d = {
            1: 'M',
            2: 'F',
        }
        t1 = d[tup[0]]
        t2 = tup[1]
        return '%s%s' % (t1, t2)

      xticks = np.concatenate(
          [
              group_positions + i * bar_spacing
              for i in range(len(policy_name_list))
          ]
      )
      xlabels = np.concatenate(
          [
              list(map(nice_join, group_definitions))
              for _ in range(len(policy_name_list))
          ]
      )
      plt.xticks(xticks, xlabels, rotation=90)

      if alpha == 0:
        s = 'A1c < 8'
      elif alpha == 1.0:
        s = 'Active'
      else:
        s = '?'

      plt.ylabel('%s %s %s Per Group' % (REWARDS_AGGREGATION_METHOD[1], '%', s))

      time_delta = horizon
      if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
        time_delta = REWARDS_AGGREGATION_METHOD[2]
      plt.title(
          r'Reward Over %s Months, $\alpha$:%.2f' % (time_delta, alpha)
          + '\n'
          + '(%s Patients, %s Monthly Interventions)' % (n_arms, budget)
      )

      plt.legend(ncol=2, loc='lower center')
      plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2, right=0.95)

      sub_dir_name = 'key_plots'
      sub_dir_path = os.path.join(img_dir, sub_dir_name)
      if not os.path.exists(sub_dir_path):
        os.makedirs(sub_dir_path)

      save_str = '%s/group_health_bar_chart_seed-%s_ntrials-%s_narms%s_all'
      save_str += '_b%.2f_alpha%.2f_%s.%s'
      plt.savefig(
          save_str
          % (
              sub_dir_path,
              seed,
              n_trials,
              n_arms,
              budget_frac,
              alpha,
              REWARDS_AGGREGATION_METHOD,
              IMAGE_TYPE,
          ),
          dpi=200,
      )
      plt.clf()
      fig = plt.gcf()
      plt.close(fig)

      if csv:
        sub_dir_path = util.Config.get_csv_summary_output_path(expname)

        fname = '%s/group_reward_chart_seed-%s_ntrials-%s_narms%s_budget%s'
        fname += '_alpha%.2f_%s.csv'
        fname_means = fname % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
            'means',
        )
        fname_errors = fname % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
            'errors',
        )

        columns = list(map(str, np.arange(n_groups + 1)))
        pretty_policy_name_list = [
            policy_name_conversion[policy_name]
            for policy_name in policy_name_list
        ]
        df_means = pd.DataFrame(
            rewards_means, index=pretty_policy_name_list, columns=columns
        )
        df_errors = pd.DataFrame(
            rewards_errors, index=pretty_policy_name_list, columns=columns
        )

        df_means.to_csv(fname_means)
        df_errors.to_csv(fname_errors)

        fname = '%s/group_mad_scores_seed-%s_ntrials-%s_narms%s_budget%s'
        fname += '_alpha%.2f.csv'
        fname_mads = fname % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            alpha,
        )

        pretty_policy_name_list = [
            policy_name_conversion[policy_name]
            for policy_name in policy_name_list
        ]
        df_mads = pd.DataFrame(
            mad_scores.reshape(1, -1), columns=pretty_policy_name_list
        )

        df_mads.to_csv(fname_mads, index=False)


def capacity_planning_plot(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alpha_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Capacity planning plot."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=small_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  # policy_name_list_pretty = [
  #     policy_name_conversion[pname] for pname in policy_name_list
  # ]

  n_budgets = len(budget_frac_list)
  n_policies = len(policy_name_list)
  n_alphas = len(alpha_list)

  rewards_means = np.zeros((n_policies, n_alphas, n_budgets))
  rewards_errors = np.zeros((n_policies, n_alphas, n_budgets))

  # alpha_colors = ['b', 'r', 'g', 'm']
  # policy_styles = [':', '-']

  for k, alpha in enumerate(alpha_list):
    for i, policy_name in enumerate(policy_name_list):
      for j, budget_frac in enumerate(budget_frac_list):
        # budget = int(n_arms * budget_frac)

        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']

        data = problem_pickle['data']

        reward_list = data[policy_name]['rewards']

        state_list = data[policy_name][
            'states'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        reward_list_accounted = reward_accounting(
            reward_list,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )
        reward_list = reward_list_accounted

        reward_list = reward_list.sum(axis=-1)  # sum over arms

        rewards_means[i, k, j] = reward_list.mean()
        rewards_errors[i, k, j] = reward_list.std() / np.sqrt(n_trials)

      xpos = np.array(budget_frac_list) * n_arms
      plt.errorbar(
          xpos,
          rewards_means[i, k],
          yerr=rewards_errors[i, k],
          color=policy_colors[policy_name],
      )

    pos_legendspace = 40
    neg_delta = 10
    plt.ylim(
        rewards_means[:, k].min() - neg_delta,
        rewards_means[:, k].max() + pos_legendspace,
    )

    if alpha == 0:
      s = 'A1c<8'
      target = 200
    elif alpha == 1.0:
      s = 'Active'
      target = 150
    else:
      target = 200
      s = r'(Weighted, $\alpha=%.2f$)' % alpha

    plt.ylabel('%s # Patients %s' % (REWARDS_AGGREGATION_METHOD[1], s))
    plt.xlabel('Monthly Intervention Budget')

    plt.plot(
        [min(budget_frac_list) * n_arms, max(budget_frac_list) * n_arms],
        [target, target],
        linestyle='--',
        color='k',
    )

    handles = []

    for policy_name in policy_name_list[::-1]:
      newhandle = Line2D(
          [0],
          [0],
          label='%s' % policy_name_conversion[policy_name],
          color=policy_colors[policy_name],
      )
      handles.append(newhandle)

    newhandle = Line2D([0], [0], label='Target', color='k', linestyle='--')
    handles.append(newhandle)

    plt.legend(handles=handles, ncol=2)
    plt.subplots_adjust(bottom=0.15, left=0.15)

    sub_dir_name = 'key_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    plt.savefig(
        '%s/capacity_planning_seed-%s_ntrials-%s_narms%s_alpha%.2f_all_%s.%s'
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            alpha,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        )
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/capacity_planning_seed-%s_ntrials-%s_narms%s_alpha%.2f_%s.csv'
      fname_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          alpha,
          'means',
      )
      fname_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          alpha,
          'errors',
      )

      columns = list(map(str, np.array(budget_frac_list) * n_arms))
      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      df_means = pd.DataFrame(
          rewards_means[:, k], index=pretty_policy_name_list, columns=columns
      )
      df_errors = pd.DataFrame(
          rewards_errors[:, k], index=pretty_policy_name_list, columns=columns
      )

      df_means.to_csv(fname_means)
      df_errors.to_csv(fname_errors)


def health_engagement_pareto_plot(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Health engagement Pareto plot."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  # policy_name_list_pretty = [
  #     policy_name_conversion[pname] for pname in policy_name_list
  # ]

  n_alphas = len(alphas_list)
  n_budgets = len(budget_frac_list)

  # make alpha pareto plot
  eng_reward_means = np.zeros((n_policies, n_budgets, n_alphas))
  eng_reward_errors = np.zeros((n_policies, n_budgets, n_alphas))
  health_reward_means = np.zeros((n_policies, n_budgets, n_alphas))
  health_reward_errors = np.zeros((n_policies, n_budgets, n_alphas))

  for i, budget_frac in enumerate(budget_frac_list):
    budget = int(budget_frac * n_arms)

    for j, alpha in enumerate(alphas_list):
      for k, policy_name in enumerate(policy_name_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']
        rewards_2d = env_info['rewards_2d']
        data = problem_pickle['data']

        state_list = data[policy_name]['states']
        eng_rewards = state_to_reward(state_list, rewards=rewards_2d, ind=0)
        health_rewards = state_to_reward(state_list, rewards=rewards_2d, ind=1)

        state_list = data[policy_name][
            'states'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        eng_reward_list_accounted = reward_accounting(
            eng_rewards,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )
        health_reward_list_accounted = reward_accounting(
            health_rewards,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )

        eng_rewards_arm_sum = eng_reward_list_accounted.sum(axis=-1)
        health_rewards_arm_sum = health_reward_list_accounted.sum(axis=-1)

        # compute stats
        eng_reward_means[k, i, j] = np.mean(eng_rewards_arm_sum)
        eng_reward_errors[k, i, j] = np.std(eng_rewards_arm_sum) / np.sqrt(
            n_trials
        )

        health_reward_means[k, i, j] = np.mean(health_rewards_arm_sum)
        health_reward_errors[k, i, j] = np.std(
            health_rewards_arm_sum
        ) / np.sqrt(n_trials)

    for k, policy_name in enumerate(policy_name_list):
      plt.errorbar(
          eng_reward_means[k, i],
          health_reward_means[k, i],
          xerr=eng_reward_errors[k, i],
          yerr=health_reward_errors[k, i],
          label=policy_name_conversion[policy_name],
          color=policy_colors[policy_name],
      )
      if policy_name == WhittleIndexPolicy.policy_key():
        eps = 0.1
        for j, alpha in enumerate(alphas_list):
          # if j%2==0:
          plt.text(
              eng_reward_means[k, i, j] + eps,
              health_reward_means[k, i, j] + eps,
              r'$\alpha=%s$' % alpha,
          )

    plt.xlabel('%s # Patients Active' % (REWARDS_AGGREGATION_METHOD[1]))
    plt.ylabel('%s # Patients A1c<8' % (REWARDS_AGGREGATION_METHOD[1]))
    plt.legend(loc='lower center')
    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2, right=0.9)

    # time_delta = horizon
    # if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
    #   time_delta = REWARDS_AGGREGATION_METHOD[2]

    sub_dir_name = 'key_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    plt.savefig(
        '%s/key_eng_v_health_pareto_seed-%s_ntrials-%s_narms%s_budget%s_%s.%s'
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        )
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = (
          '%s/eng_v_health_pareto_seed-%s_ntrials-%s_narms%s_budget%s_%s_%s.csv'
      )
      fname_eng_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'eng',
          'means',
      )
      fname_eng_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'eng',
          'errors',
      )
      fname_health_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'health',
          'means',
      )
      fname_health_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'health',
          'errors',
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      columns = list(map(str, alphas_list))
      df_eng_means = pd.DataFrame(
          eng_reward_means[:, i], index=pretty_policy_name_list, columns=columns
      )
      df_eng_errors = pd.DataFrame(
          eng_reward_errors[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )

      df_health_means = pd.DataFrame(
          health_reward_means[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )
      df_health_errors = pd.DataFrame(
          health_reward_errors[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )

      df_eng_means.to_csv(fname_eng_means)
      df_eng_errors.to_csv(fname_eng_errors)
      df_health_means.to_csv(fname_health_means)
      df_health_errors.to_csv(fname_health_errors)


def fairness_pareto_plots(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    alphas_list,
    budget_frac_list,
    img_dir,
    policy_name_list=None,
    expname=None,
    alpha_fairness=True,
    environment_class=None,
):
  """Fairness Pareto plots."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=small_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  # policy_name_list = [
  #     NoActionPolicy.policy_key(),
  #     HighA1cRRPolicy.policy_key(),
  #     RandomHighA1cPolicy.policy_key(),
  #     WhittleIndexPolicy.policy_key(),
  #     EquitableLPPolicyMMR.policy_key(),
  #     EquitableLPPolicyMNW.policy_key(),
  # ]
  n_policies = len(policy_name_list)

  # policy_name_list_pretty = [
  #     policy_name_conversion[pname] for pname in policy_name_list
  # ]

  n_alphas = len(alphas_list)
  n_budgets = len(budget_frac_list)

  _, group_mappings, _, _ = (
      environment_class.parse_model_data(args.model, n_arms)
  )
  group_list = sorted(list(set(group_mappings)))
  n_groups = len(group_list)

  # make alpha pareto plot
  eng_reward_means = np.zeros((n_policies, n_budgets, n_alphas))
  eng_reward_errors = np.zeros((n_policies, n_budgets, n_alphas))
  health_reward_means = np.zeros((n_policies, n_budgets, n_alphas))
  health_reward_errors = np.zeros((n_policies, n_budgets, n_alphas))
  group_rewards_means = np.zeros((n_policies, n_budgets, n_alphas, n_groups))
  group_rewards_errors = np.zeros((n_policies, n_budgets, n_alphas, n_groups))
  fairness_means = np.zeros((n_policies, n_budgets, n_alphas))
  fairness_errors = np.zeros((n_policies, n_budgets, n_alphas))

  if not alpha_fairness:
    group_eng_rewards_means = np.zeros(
        (n_policies, n_budgets, n_alphas, n_groups)
    )
    group_eng_rewards_errors = np.zeros(
        (n_policies, n_budgets, n_alphas, n_groups)
    )
    fairness_eng_means = np.zeros((n_policies, n_budgets, n_alphas))
    fairness_eng_errors = np.zeros((n_policies, n_budgets, n_alphas))
    group_health_rewards_means = np.zeros(
        (n_policies, n_budgets, n_alphas, n_groups)
    )
    group_health_rewards_errors = np.zeros(
        (n_policies, n_budgets, n_alphas, n_groups)
    )
    fairness_health_means = np.zeros((n_policies, n_budgets, n_alphas))
    fairness_health_errors = np.zeros((n_policies, n_budgets, n_alphas))

  for i, budget_frac in enumerate(budget_frac_list):
    budget = int(budget_frac * n_arms)

    for j, alpha in enumerate(alphas_list):
      for k, policy_name in enumerate(policy_name_list):
        file_name = input_file_name_template % (
            expname,
            n_arms,
            budget_frac,
            environment_name,
            n_trials,
            seed,
            horizon,
            alpha,
        )

        problem_pickle = {}
        with open(file_name, 'rb') as fh:
          problem_pickle = pickle.load(fh)

        # environment_parameters = problem_pickle['simulation_parameters'][
        #     'environment_parameters'
        # ]

        env_info = problem_pickle['simulation_parameters']['environment_info']
        rewards_2d = env_info['rewards_2d']
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        data = problem_pickle['data']

        state_list = data[policy_name]['states']
        state_index_to_tuple_dict = env_info['state_index_to_tuple']

        eng_do_sc_trajectories = state_ind_eng_do_sc_combined(
            state_list, state_index_to_tuple_dict
        )
        health_rewards = state_to_reward(state_list, rewards=rewards_2d, ind=1)

        # all remaining 1s represent states that were engaged
        engaged_or_maint_trajectories = np.copy(eng_do_sc_trajectories)
        # change maints to engs
        engaged_or_maint_trajectories[engaged_or_maint_trajectories == 0] = 1
        # zero out others
        engaged_or_maint_trajectories[engaged_or_maint_trajectories != 1] = 0

        eng_reward_list_accounted = reward_accounting(
            engaged_or_maint_trajectories,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )
        health_reward_list_accounted = reward_accounting(
            health_rewards,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )

        eng_rewards_all_arms = eng_reward_list_accounted.sum(axis=-1)
        health_rewards_all_arms = health_reward_list_accounted.sum(axis=-1)

        # compute stats
        eng_reward_means[k, i, j] = np.mean(eng_rewards_all_arms)
        eng_reward_errors[k, i, j] = np.std(eng_rewards_all_arms) / np.sqrt(
            n_trials
        )

        health_reward_means[k, i, j] = np.mean(health_rewards_all_arms)
        health_reward_errors[k, i, j] = np.std(
            health_rewards_all_arms
        ) / np.sqrt(n_trials)

        reward_list = data[policy_name][
            'rewards'
        ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
        reward_list_accounted = reward_accounting(
            reward_list,
            state_list,
            REWARDS_AGGREGATION_METHOD,
            state_index_to_tuple_dict,
        )

        if alpha_fairness:
          for group in group_list:
            group_inds = group_mappings == group
            group_size = group_inds.sum()
            total_rewards_group = (
                reward_list_accounted[:, group_inds].sum(axis=-1) / group_size
            )  # mean over arms in group
            group_rewards_means[k, i, j, group] = (
                total_rewards_group.mean()
            )  # mean over trials
            group_rewards_errors[
                k, i, j, group
            ] = total_rewards_group.std() / np.sqrt(
                n_trials
            )  # std over trials

          mad_num = 0
          mad_error_num = 0
          for x in group_list:
            for y in group_list:
              abs_diff = abs(
                  group_rewards_means[k, i, j, x]
                  - group_rewards_means[k, i, j, y]
              )
              mad_num += abs_diff
              mad_error_num += (
                  group_rewards_errors[k, i, j, x] ** 2
                  + group_rewards_errors[k, i, j, y] ** 2
              )

          mad_error_num = np.sqrt(mad_error_num)

          groups_mean = group_rewards_means[k, i, j].mean()
          mad = mad_num / (2 * n_groups**2 * groups_mean)  # Gini coefficient

          mad_error_denom = np.sqrt(
              np.sum([x**2 for x in group_rewards_errors[k, i, j]])
          )
          mad_error = mad * np.sqrt(
              mad_error_num / mad_num + mad_error_denom / groups_mean
          )

          fairness_means[k, i, j] = mad
          fairness_errors[k, i, j] = mad_error

        else:
          # compute health and fairness rewards
          for group in group_list:
            group_inds = group_mappings == group
            group_size = group_inds.sum()

            eng_rewards_group = reward_accounting(
                engaged_or_maint_trajectories,
                state_list,
                REWARDS_AGGREGATION_METHOD,
                state_index_to_tuple_dict,
            )
            eng_rewards_group = (
                eng_rewards_group[:, group_inds].sum(axis=-1) / group_size
            )
            health_rewards_group = reward_accounting(
                health_rewards,
                state_list,
                REWARDS_AGGREGATION_METHOD,
                state_index_to_tuple_dict,
            )
            health_rewards_group = (
                health_rewards_group[:, group_inds].sum(axis=-1) / group_size
            )

            group_eng_rewards_means[k, i, j, group] = (
                eng_rewards_group.mean()
            )  # mean over trials
            group_eng_rewards_errors[k, i, j, group] = (
                eng_rewards_group.std() / np.sqrt(n_trials)
            )

            group_health_rewards_means[k, i, j, group] = (
                health_rewards_group.mean()
            )  # mean over trials
            group_health_rewards_errors[k, i, j, group] = (
                health_rewards_group.std() / np.sqrt(n_trials)
            )

          eng_mad_num = 0
          eng_mad_error_num = 0
          health_mad_num = 0
          health_mad_error_num = 0
          for x in group_list:
            for y in group_list:
              eng_abs_diff = abs(
                  group_eng_rewards_means[k, i, j, x]
                  - group_eng_rewards_means[k, i, j, y]
              )
              eng_mad_num += eng_abs_diff
              eng_mad_error_num += (
                  group_eng_rewards_errors[k, i, j, x] ** 2
                  + group_eng_rewards_errors[k, i, j, y] ** 2
              )

              health_abs_diff = abs(
                  group_health_rewards_means[k, i, j, x]
                  - group_health_rewards_means[k, i, j, y]
              )
              health_mad_num += health_abs_diff
              health_mad_error_num += (
                  group_health_rewards_errors[k, i, j, x] ** 2
                  + group_health_rewards_errors[k, i, j, y] ** 2
              )

          eng_mad_error_num = np.sqrt(eng_mad_error_num)
          health_mad_error_num = np.sqrt(health_mad_error_num)

          eng_groups_mean = group_eng_rewards_means[k, i, j].mean()
          eng_mad = eng_mad_num / (
              2 * n_groups**2 * eng_groups_mean
          )  # Gini coefficient

          eng_mad_error_denom = np.sqrt(
              np.sum([x**2 for x in group_eng_rewards_errors[k, i, j]])
          )
          eng_mad_error = eng_mad * np.sqrt(
              eng_mad_error_num / eng_mad_num
              + eng_mad_error_denom / eng_groups_mean
          )

          fairness_eng_means[k, i, j] = eng_mad
          fairness_eng_errors[k, i, j] = eng_mad_error

          health_groups_mean = group_health_rewards_means[k, i, j].mean()
          health_mad = health_mad_num / (2 * n_groups**2 * health_groups_mean)

          health_mad_error_denom = np.sqrt(
              np.sum([x**2 for x in group_health_rewards_errors[k, i, j]])
          )
          health_mad_error = health_mad * np.sqrt(
              health_mad_error_num / health_mad_num
              + health_mad_error_denom / health_groups_mean
          )

          fairness_health_means[k, i, j] = health_mad
          fairness_health_errors[k, i, j] = health_mad_error

    # fairness vs. eng rewards
    for k, policy_name in enumerate(policy_name_list):
      y_means = None
      y_errs = None
      if alpha_fairness:
        y_means = fairness_means
        y_errs = fairness_errors
      else:
        y_means = fairness_eng_means
        y_errs = fairness_eng_errors

      plt.errorbar(
          eng_reward_means[k, i],
          y_means[k, i],
          xerr=eng_reward_errors[k, i],
          yerr=y_errs[k, i],
          label=policy_name_conversion[policy_name],
          color=policy_colors[policy_name],
      )
      if policy_name in [
          WhittleIndexPolicy.policy_key(),
          EquitableLPPolicyMMR.policy_key(),
      ]:
        eps = 0.000
        for j, alpha in enumerate(alphas_list):
          if j % 2 == 0:
            plt.text(
                eng_reward_means[k, i, j] + eps,
                y_means[k, i, j] + eps,
                '%s' % alpha,
            )

    # plt.xlabel('# Engaged Patient-Months')
    plt.xlabel('%s # Patients Active' % (REWARDS_AGGREGATION_METHOD[1]))
    plt.ylabel('Gini Coefficient')  # \n (MAD of Avg. Group Rewards)')
    plt.legend()
    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2)

    time_delta = horizon
    if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
      time_delta = REWARDS_AGGREGATION_METHOD[2]

    plt.title(
        'Fairness vs. Engagement \n (%s Months, %s Patients, %s Budget)'
        % (time_delta, n_arms, budget)
    )

    sub_dir_name = 'key_plots'
    if not alpha_fairness:
      sub_dir_name = 'extra_fairness_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    plt.savefig(
        '%s/key_fairness_v_eng_pareto_seed-%s_ntrials-%s_narms%s_budget%s_%s.%s'
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        )
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    # fairness vs. health rewards
    for k, policy_name in enumerate(policy_name_list):
      y_means = None
      y_errs = None
      if alpha_fairness:
        y_means = fairness_means
        y_errs = fairness_errors
      else:
        y_means = fairness_health_means
        y_errs = fairness_health_errors
      plt.errorbar(
          health_reward_means[k, i],
          y_means[k, i],
          xerr=health_reward_errors[k, i],
          yerr=y_errs[k, i],
          label=policy_name_conversion[policy_name],
          color=policy_colors[policy_name],
      )
      if policy_name in [
          WhittleIndexPolicy.policy_key(),
          EquitableLPPolicyMMR.policy_key(),
      ]:
        epsy = 0.00
        epsx = 0
        for j, alpha in enumerate(alphas_list):
          if j % 2 == 0:
            plt.text(
                health_reward_means[k, i, j] + epsx,
                y_means[k, i, j] + epsy,
                '%s' % alpha,
            )
            epsx = 0

    plt.xlabel('%s # Patients A1c<8' % (REWARDS_AGGREGATION_METHOD[1]))
    plt.ylabel('Gini Coefficient')  # \n (MAD of Avg. Group Rewards)')
    plt.legend()
    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.2)

    plt.title(
        'Fairness vs. Health \n (%s Months, %s Patients, %s Budget)'
        % (time_delta, n_arms, budget)
    )

    sub_dir_name = 'key_plots'
    if not alpha_fairness:
      sub_dir_name = 'extra_fairness_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    save_str = '%s/key_fairness_v_health_pareto_seed-%s_ntrials-%s_narms%s'
    save_str += '_budget%s_%s.%s'
    plt.savefig(
        save_str
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        )
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/pareto_data_seed-%s_ntrials-%s_narms%s_budget%s_%s_%s.csv'
      fname_eng_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'eng',
          'means',
      )
      fname_eng_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'eng',
          'errors',
      )
      fname_health_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'health',
          'means',
      )
      fname_health_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'health',
          'errors',
      )
      fname_fairness_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness',
          'means',
      )
      fname_fairness_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness',
          'errors',
      )

      fname_fairness_health_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness_health',
          'means',
      )
      fname_fairness_health_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness_health',
          'errors',
      )
      fname_fairness_eng_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness_eng',
          'means',
      )
      fname_fairness_eng_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'fairness_eng',
          'errors',
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      columns = list(map(str, alphas_list))
      df_eng_means = pd.DataFrame(
          eng_reward_means[:, i], index=pretty_policy_name_list, columns=columns
      )
      df_eng_errors = pd.DataFrame(
          eng_reward_errors[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )

      df_health_means = pd.DataFrame(
          health_reward_means[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )
      df_health_errors = pd.DataFrame(
          health_reward_errors[:, i],
          index=pretty_policy_name_list,
          columns=columns,
      )

      if alpha_fairness:
        df_fairness_means = pd.DataFrame(
            fairness_means[:, i], index=pretty_policy_name_list, columns=columns
        )
        df_fairness_errors = pd.DataFrame(
            fairness_errors[:, i],
            index=pretty_policy_name_list,
            columns=columns,
        )

      else:
        df_fairness_health_means = pd.DataFrame(
            fairness_health_means[:, i],
            index=pretty_policy_name_list,
            columns=columns,
        )
        df_fairness_health_errors = pd.DataFrame(
            fairness_health_errors[:, i],
            index=pretty_policy_name_list,
            columns=columns,
        )

        df_fairness_eng_means = pd.DataFrame(
            fairness_eng_means[:, i],
            index=pretty_policy_name_list,
            columns=columns,
        )
        df_fairness_eng_errors = pd.DataFrame(
            fairness_eng_errors[:, i],
            index=pretty_policy_name_list,
            columns=columns,
        )

      df_eng_means.to_csv(fname_eng_means)
      df_eng_errors.to_csv(fname_eng_errors)
      df_health_means.to_csv(fname_health_means)
      df_health_errors.to_csv(fname_health_errors)

      if alpha_fairness:
        df_fairness_means.to_csv(fname_fairness_means)
        df_fairness_errors.to_csv(fname_fairness_errors)

      else:
        df_fairness_health_means.to_csv(fname_fairness_health_means)
        df_fairness_health_errors.to_csv(fname_fairness_health_errors)
        df_fairness_eng_means.to_csv(fname_fairness_eng_means)
        df_fairness_eng_errors.to_csv(fname_fairness_eng_errors)



def group_reward_difference_bar_chart_general(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    budget_fracs,
    img_dir,
    expname,
    policy_name_list=None,
    environment_class=None,
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
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

#   plt.rcParams['figure.figsize'] = (8, 3)
  

  # policy_name_list = [
  #     EquitableLPPolicyMMR.policy_key(),
  #     EquitableLPPolicyMMR.policy_key(),
  # ]
  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  _, group_mappings, _ = (
      environment_class.parse_model_data(args.model, n_arms)
  )
  group_list = sorted(list(set(group_mappings)))
  n_groups = len(group_list)

  rewards_means = np.zeros((n_policies, n_groups + 1))
  rewards_errors = np.zeros((n_policies, n_groups + 1))

#   barwidth = 0.14
  barwidth = 0.18

#   group_positions = np.array([
#       -1.5 * barwidth,
#       -0.5 * barwidth,
#       0.5 * barwidth,
#       1.5 * barwidth,
#       2.5 * barwidth,
#   ])
  group_positions = np.arange(n_groups)+1
  group_positions = group_positions - group_positions.mean()
  group_positions = group_positions*barwidth
  bar_spacing = 1

  fig, axs = plt.subplots(len(budget_fracs), 1, figsize=(8, 4))

  for b, budget_frac in enumerate(budget_fracs):
    budget = int(n_arms * budget_frac)

    mad_scores = np.zeros(n_policies)
    for i, policy_name in enumerate(policy_name_list):
      file_name = input_file_name_template % (
          expname,
          n_arms,
          budget_frac,
          environment_name,
          n_trials,
          seed,
          horizon,
      )

      problem_pickle = {}
      with open(file_name, 'rb') as fh:
        problem_pickle = pickle.load(fh)

      env_info = problem_pickle['simulation_parameters']['environment_info']

      data = problem_pickle['data']

      reward_list = data[policy_name][
          'rewards'
      ]
      state_list = data[policy_name][
          'states'
      ]

      reward_list_accounted = reward_accounting(
          reward_list,
          state_list,
          REWARDS_AGGREGATION_METHOD,
          None,
      )

      for group in group_list:
        group_inds = group_mappings == group
        group_size = group_inds.sum()
        rewards_group = (
            reward_list_accounted[:, group_inds].sum(axis=-1) / group_size
        )  # sum over arms in group
        rewards_means[i, group] = rewards_group.mean()
        rewards_errors[i, group] = (
            rewards_group.std() / np.sqrt(n_trials)
        )

      arm_sum = reward_list_accounted.sum(
          axis=-1
      )  # sum over arms in g1
      rewards_means[i, n_groups] = arm_sum.mean() / n_arms
      rewards_errors[i, n_groups] = (
          arm_sum.std() / n_arms / np.sqrt(n_trials)
      )

      mad = 0
      for j in group_list:
        for k in group_list:
          mad += abs(rewards_means[i, j] - rewards_means[i, k])
      mad /= 2 * n_groups**2 * rewards_means[i].mean()

      axs[b].bar(
          group_positions + i * bar_spacing,
          rewards_means[i, :-1],
          yerr=rewards_errors[i, :-1],
          align='center',
          alpha=0.8,
          width=barwidth,
          edgecolor='black',
          ecolor='black',
          capsize=2,
          color=policy_colors[policy_name],
          label=policy_name_list_pretty[i]  # + ', Gini=%.3f' % mad,
      )
      mad_scores[i] = mad

    x_pos = 0+0.5
    y_pos = rewards_means.max()*0.8
    axs[b].text(x_pos, y_pos, 'By Group', horizontalalignment='center')

    x_pos = n_policies + 1
    # y_pos = rewards_means[:,-1].max()*1.5
    axs[b].text(x_pos, y_pos, 'Total', horizontalalignment='center')

    # add one more spot for the arm means
    pol_mean_positions = np.arange(n_policies)
    pol_mean_positions = pol_mean_positions - pol_mean_positions.mean()
    pol_mean_positions = pol_mean_positions*barwidth
    bar_spacing = 1
    more_space = 1*bar_spacing
    for i, policy_name in enumerate(policy_name_list):
        axs[b].bar(
            pol_mean_positions[i] + n_policies * bar_spacing + more_space,
            rewards_means[i, -1],
            yerr=rewards_errors[i, -1],
            align='center',
            alpha=0.8,
            width=barwidth,
            edgecolor='black',
            ecolor='black',
            capsize=2,
            hatch='/',
            color=policy_colors[policy_name],
            #   label=policy_name_list_pretty[i] + ' %.3f' % mad,
        )

    xticks = np.concatenate(
        [
            group_positions + i * bar_spacing
            for i in range(n_policies)
        ] + [pol_mean_positions + n_policies*bar_spacing + more_space]
    )
    xlabels=[]
    if b == len(budget_fracs) - 1:
      xlabels = np.concatenate(
          [
              list(map(str, group_list))
            #   ['A','B','C','D','E']
              for _ in range(n_policies)
          ] + [['' for _ in range(n_policies)]]
      )
    # plt.xticks(xticks, xlabels, rotation=90)
    axs[b].set_xticks(xticks, xlabels)

    axs[b].set_yticks(np.arange(0, rewards_means.max(), step=4))  # Set label locations.
    axs[b].grid(axis='y')

    x = (n_policies) * bar_spacing
    maxval = rewards_means.max()
    axs[b].plot([x,x],[0,maxval],linestyle='--')

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/group_reward_chart_seed-%s_ntrials-%s_narms%s_budget%s'
      fname += '_%s.csv'
      fname_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'means',
      )
      fname_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'errors',
      )

      columns = list(map(str, np.arange(n_groups + 1)))
      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      df_means = pd.DataFrame(
          rewards_means, index=pretty_policy_name_list, columns=columns
      )
      df_errors = pd.DataFrame(
          rewards_errors, index=pretty_policy_name_list, columns=columns
      )

      df_means.to_csv(fname_means)
      df_errors.to_csv(fname_errors)

      fname = '%s/group_mad_scores_seed-%s_ntrials-%s_narms%s_budget%s'
      fname += '.csv'
      fname_mads = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      df_mads = pd.DataFrame(
          mad_scores.reshape(1, -1), columns=pretty_policy_name_list
      )

      df_mads.to_csv(fname_mads, index=False)
  

  # fig.supylabel('Mean %s Reward' % REWARDS_AGGREGATION_METHOD[1])
  axs[0].set_ylabel('Mean %s Reward' % REWARDS_AGGREGATION_METHOD[1])
  axs[0].yaxis.set_label_coords(-.1, -.1)


  time_delta = horizon
  if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
    time_delta = REWARDS_AGGREGATION_METHOD[2]
  # plt.title(
  #     r'Reward Over %s Step' % time_delta
  #     + '\n'
  #     + '(%s Arms, %s Actions Per Steps)' % (n_arms, budget)
  # )
  
  axs[0].legend(ncol=n_policies, loc='upper center', bbox_to_anchor=(0.44, 1.5))
  fig.subplots_adjust(top=0.75, left=0.2, bottom=0.1, right=0.95)

  # fig.yaxis.set_label_coords(-.5, 0)

  sub_dir_name = 'key_plots'
  sub_dir_path = os.path.join(img_dir, sub_dir_name)
  if not os.path.exists(sub_dir_path):
    os.makedirs(sub_dir_path)

  save_str = '%s/group_reward_bar_chart_seed-%s_ntrials-%s_narms%s_all'
  save_str += '_b%.2f_%s.%s'
  plt.savefig(
      save_str
      % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget_frac,
          REWARDS_AGGREGATION_METHOD,
          IMAGE_TYPE,
      ),
      dpi=200,
  )
  plt.clf()
  # fig = plt.gcf()
  plt.close(fig)


def health_difference_bar_chart_general(
    n_arms,
    seed,
    n_trials,
    horizon,
    environment_name,
    input_file_name_template,
    budget_fracs,
    img_dir,
    policy_name_list=None,
    expname=None,
):
  """Health difference bar chart."""

  small_size = 14
  medium_size = 18
  bigger_size = 22

  plt.rc('font', size=medium_size)  # controls default text sizes
  plt.rc('axes', titlesize=medium_size)  # fontsize of the axes title
  plt.rc('axes', labelsize=medium_size)  # fontsize of the x and y labels
  plt.rc('xtick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('ytick', labelsize=medium_size)  # fontsize of the tick labels
  plt.rc('legend', fontsize=small_size)  # legend fontsize
  plt.rc('figure', titlesize=bigger_size)  # fontsize of the figure title

  n_policies = len(policy_name_list)

  policy_name_list_pretty = [
      policy_name_conversion[pname] for pname in policy_name_list
  ]

  rewards_means = np.zeros((n_policies))
  rewards_errors = np.zeros((n_policies))

  for budget_frac in budget_fracs:
    budget = int(n_arms * budget_frac)
    for i, policy_name in enumerate(policy_name_list):
      file_name = input_file_name_template % (
          expname,
          n_arms,
          budget_frac,
          environment_name,
          n_trials,
          seed,
          horizon,
      )

      problem_pickle = {}
      with open(file_name, 'rb') as fh:
        problem_pickle = pickle.load(fh)

      # environment_parameters = problem_pickle['simulation_parameters'][
      #     'environment_parameters'
      # ]

      env_info = problem_pickle['simulation_parameters']['environment_info']

      data = problem_pickle['data']

      reward_list = data[policy_name][
          'rewards'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)
      state_list = data[policy_name][
          'states'
      ]  # when alpha=0 rewards = (Expected # users w/ A1c < 8)

      reward_list_accounted = reward_accounting(
          reward_list,
          state_list,
          REWARDS_AGGREGATION_METHOD,
          None,
      )

      reward_arm_sum = reward_list_accounted.sum(axis=-1)  # sum over arms
      rewards_means[i] = reward_arm_sum.mean()
      rewards_errors[i] = reward_arm_sum.std() / np.sqrt(n_trials)

      plt.bar(
          i,
          rewards_means[i],
          yerr=rewards_errors[i],
          align='center',
          alpha=0.8,
          ecolor='black',
          capsize=10,
          color=policy_colors[policy_name],
          label=policy_name_list_pretty[i],
      )

    plt.xticks(np.arange(n_policies), policy_name_list_pretty, rotation=35)
    plt.ylabel('%s Reward' % (REWARDS_AGGREGATION_METHOD[1]))
    # plt.xlabel('Policy')
    time_delta = horizon
    if REWARDS_AGGREGATION_METHOD[0] == 'Timepoints':
      time_delta = REWARDS_AGGREGATION_METHOD[2]

    plt.title(
        r'Reward (%s months)' % (time_delta,)
        + '\n'
        + '(%s Arms, %s Budget)' % (n_arms, budget)
    )
    plt.ylim([
        rewards_means.min() - 0.05 * n_arms,
        rewards_means.max() + 0.05 * n_arms,
    ])

    sub_dir_name = 'key_plots'
    sub_dir_path = os.path.join(img_dir, sub_dir_name)
    if not os.path.exists(sub_dir_path):
      os.makedirs(sub_dir_path)

    plt.subplots_adjust(top=0.8, left=0.2, bottom=0.3)

    # plt.tight_layout()
    save_str = '%s/health_bar_chart_seed-%s_ntrials-'
    save_str += '%s_narms%s_b%.2f_all_%s.%s'
    plt.savefig(
        save_str
        % (
            sub_dir_path,
            seed,
            n_trials,
            n_arms,
            budget,
            REWARDS_AGGREGATION_METHOD,
            IMAGE_TYPE,
        ),
        dpi=200,
    )
    plt.clf()
    fig = plt.gcf()
    plt.close(fig)

    if csv:
      sub_dir_path = util.Config.get_csv_summary_output_path(expname)

      fname = '%s/health_bar_chart_seed-%s_ntrials-%s_narms%s_b%.2f_'
      fname += '%s.csv'
      fname_means = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'means',
      )
      fname_errors = fname % (
          sub_dir_path,
          seed,
          n_trials,
          n_arms,
          budget,
          'errors',
      )

      pretty_policy_name_list = [
          policy_name_conversion[policy_name]
          for policy_name in policy_name_list
      ]
      columns = ['Value']
      df_means = pd.DataFrame(
          rewards_means, index=pretty_policy_name_list, columns=columns
      )
      df_errors = pd.DataFrame(
          rewards_errors, index=pretty_policy_name_list, columns=columns
      )

      df_means.to_csv(fname_means)
      df_errors.to_csv(fname_errors)



if __name__ == '__main__':
  environment_name_in = param_dict['environment_name']

  if environment_name_in == DiabetesAppPartialObsEnvironment.env_key():
    environment_class_in = DiabetesAppPartialObsEnvironment
    env_more_params_in = environment_class_in.environment_params().keys()

    combined_filename_template_in = util.Config.get_combined_filename_template(
        environment_class_in, args.expname
    )
    combined_filename_template_in += '.pkl'

    batch_dir_path_in = util.Config.get_img_output_path(args.expname)

    n_arms_list_in = param_dict['n_arms_list']
    budget_frac_list_in = param_dict['budget_frac_list']
    alpha_list_in = param_dict['alpha']
    n_trials_in = param_dict['n_trials']
    horizon_list_in = param_dict['horizon_list']
    base_seed_in = param_dict['base_seed']
    n_arms_in = n_arms_list_in[0]
    horizon_in = horizon_list_in[0]

    ########
    # High-level analyses of final outcomes and capacity planning
    ########
    group_reward_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
        environment_class=environment_class_in,
    )

    health_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

    engagement_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

    capacity_planning_plot(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

    health_engagement_pareto_plot(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

    fairness_pareto_plots(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        alpha_fairness=True,
        policy_name_list=param_dict['policy_list'],
        environment_class=environment_class_in,
    )

    fairness_pareto_plots(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        alpha_fairness=False,
        policy_name_list=param_dict['policy_list'],
        environment_class=environment_class_in,
    )

    ########
    # Closer looks at action distributions and states over time
    ########
    get_action_distros_by_alpha_scaled_by_state_distros(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        include_memory_states=False,
        policy_name_list=param_dict['policy_list'],
    )
    get_action_distros_by_alpha_scaled_by_state_distros(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        include_memory_states=True,
        policy_name_list=param_dict['policy_list'],
    )

    get_action_distros_by_alpha_with_state_distros(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        include_memory_states=False,
        policy_name_list=param_dict['policy_list'],
    )

    get_action_distros_by_alpha_with_state_distros(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        include_memory_states=True,
        policy_name_list=param_dict['policy_list'],
    )

    get_state_distros_over_time(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )
  
  elif environment_name_in == DiabetesAppEnvironment.env_key():
    environment_class_in = DiabetesAppEnvironment
    env_more_params_in = environment_class_in.environment_params().keys()

    combined_filename_template_in = util.Config.get_combined_filename_template(
        environment_class_in, args.expname
    )
    combined_filename_template_in += '.pkl'

    batch_dir_path_in = util.Config.get_img_output_path(args.expname)

    n_arms_list_in = param_dict['n_arms_list']
    budget_frac_list_in = param_dict['budget_frac_list']
    alpha_list_in = param_dict['alpha']
    n_trials_in = param_dict['n_trials']
    horizon_list_in = param_dict['horizon_list']
    base_seed_in = param_dict['base_seed']
    n_arms_in = n_arms_list_in[0]
    horizon_in = horizon_list_in[0]

    ########
    # High-level analyses of final outcomes and capacity planning
    ########
    group_reward_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
        environment_class=environment_class_in,
    )

    health_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

    engagement_difference_bar_chart(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        alpha_list_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

  else:  # environment_name_in == TwoStateEnvironment.env_key():
    environment_class_in = environments.Config.get_env_class(environment_name_in)
    env_more_params_in = environment_class_in.environment_params().keys()

    combined_filename_template_in = util.Config.get_combined_filename_template(
        environment_class_in, args.expname
    )
    combined_filename_template_in += '.pkl'

    batch_dir_path_in = util.Config.get_img_output_path(args.expname)

    n_arms_list_in = param_dict['n_arms_list']
    budget_frac_list_in = param_dict['budget_frac_list']
    n_trials_in = param_dict['n_trials']
    horizon_list_in = param_dict['horizon_list']
    base_seed_in = param_dict['base_seed']
    n_arms_in = n_arms_list_in[0]
    horizon_in = horizon_list_in[0]

    ########
    # High-level analyses of final outcomes and capacity planning
    ########
    group_reward_difference_bar_chart_general(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
        environment_class=environment_class_in,
    )

    health_difference_bar_chart_general(
        n_arms_in,
        base_seed_in,
        n_trials_in,
        horizon_in,
        environment_name_in,
        combined_filename_template_in,
        budget_frac_list_in,
        batch_dir_path_in,
        expname=args.expname,
        policy_name_list=param_dict['policy_list'],
    )

#   else:
#     err_str = 'Analysis not implemented for env "%s"' % environment_name_in
#     raise ValueError(err_str)
