"""Combines results of a set of experiments.

Author: killian-34/jakillian
"""

import argparse
import collections
import itertools
import pickle

import environments
import numpy as np
import util

# combine experimental results to easily pass to analysis scripts (as they are)
model_help = 'Name of model data file for experiments'
params_help = 'Name of parameter file for experiments'
parser = argparse.ArgumentParser(description='Combine all experiment data.')
parser.add_argument('--expname', type=str, help='Name of experiment')
parser.add_argument('--model', type=str, help=model_help)
parser.add_argument('--params', type=str, help=params_help)
parser.add_argument('--tag', type=str, help='String added to filename.')
args = parser.parse_args()

param_dict = util.parse_parameters(args.params)

environment_name = param_dict['environment_name']
base_seed = param_dict['base_seed']
n_trials = param_dict['n_trials']
policy_list = param_dict['policy_list']

environment_name = param_dict['environment_name']
environment_class = environments.Config.get_env_class(environment_name)
env_more_params = environment_class.environment_params().keys()

env_param_lists = [param_dict[key] for key in env_more_params]

all_param_combos = list(
    itertools.product(
        param_dict['n_arms_list'],
        param_dict['budget_frac_list'],
        param_dict['horizon_list'],
        *env_param_lists,
    )
)

num_unique_combos = len(all_param_combos)

files_not_found = {policy_name: 0 for policy_name in policy_list}
for experiment_index in range(num_unique_combos):
  combo = all_param_combos[experiment_index]
  n_arms = combo[0]
  budget_frac = combo[1]
  horizon = combo[2]

  # make a parameter dict out of the remaining
  env_param_dict = collections.OrderedDict()
  for i, key in enumerate(env_more_params):
    env_param_dict[key] = combo[3+i]

  data = {}
  for policy_name in policy_list:
    data[policy_name] = {
        'states': np.zeros((n_trials, n_arms, horizon), dtype=int),
        'actions': np.zeros((n_trials, n_arms, horizon), dtype=int),
        'rewards': np.zeros((n_trials, n_arms, horizon), dtype=float),
        'next_states': np.zeros((n_trials, n_arms, horizon), dtype=int),
    }

  combined_filename = util.Config.get_combined_filename(
      environment_class,
      args.expname,
      n_arms,
      budget_frac,
      environment_name,
      n_trials,
      base_seed,
      horizon,
      env_param_dict
  )
  combined_filename += '.pkl'

  for trial_number in range(n_trials):
    for policy_name in policy_list:
      filename = util.Config.get_exp_filename(
          environment_class,
          args.expname,
          n_arms,
          budget_frac,
          environment_name,
          trial_number,
          base_seed,
          horizon,
          policy_name,
          env_param_dict,
      )
      results_filename = filename + '.pkl'
      meta_filename = filename + '_meta.pkl'

      try:
        with open(results_filename, 'rb') as fo:
          data_i = pickle.load(fo)
          data[policy_name]['states'][trial_number] = data_i['states']
          data[policy_name]['actions'][trial_number] = data_i['actions']
          data[policy_name]['rewards'][trial_number] = data_i['rewards']
          data[policy_name]['next_states'][trial_number] = data_i['next_states']
      except FileNotFoundError:
        # Make a copy of previous data if missing, but raise warning
        # should only get used if some experiment is still running, but want
        # to get a quick look at plots
        print("Couldn't find file:", results_filename)
        print('Combo:', experiment_index)
        files_not_found[policy_name] += 1

        data[policy_name]['states'][trial_number] = data_i['states']
        data[policy_name]['actions'][trial_number] = data_i['actions']
        data[policy_name]['rewards'][trial_number] = data_i['rewards']
        data[policy_name]['next_states'][trial_number] = data_i['next_states']

      # meta files save all the policy data, so take the first one
      if trial_number == 0:
        with open(meta_filename, 'rb') as fo:
          simulation_parameters = pickle.load(fo)

  with open(combined_filename, 'wb') as fo:
    combined_data = {
        'data': data,
        'simulation_parameters': simulation_parameters
    }
    pickle.dump(combined_data, fo)

print('files not found')
for policy_name in policy_list:
  print(policy_name, files_not_found[policy_name])

print()

