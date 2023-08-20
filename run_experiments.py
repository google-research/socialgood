"""Run experiments for all combinations of parameters.

  This is the main executable file which runs experiments.

  Assigns a unique index to all experiment paramter combinations, 
  then will accept a range of indexes and execute those
  corresponding experiments. This enables simple parallel 
  execution of batches of experiments via an external scheduler.

  Author: killian-34/jakillian
"""

import argparse
import collections
import itertools

import environments
import numpy as np
from simulator import run_simulation
import util

parser_help = 'Setup and run parallel experiments.'
n_chunks_help = 'Number of unique experiment chunks. If set, will not run jobs,'
n_chunks_help += ' only print commands needed.'
start_help = 'Index of first job (inclusive). If set will run jobs.'
end_help = 'Index of last job (exclusive). If set will run jobs.'
expname_help = 'Name of experiment'
model_help = 'Name of model data file for experiments'
params_help = 'Name of parameter file for experiments'
test_help = 'Run a test of the repository.'
reset_help = 'How often to reset the environment (online).'
parser = argparse.ArgumentParser(description=parser_help)
parser.add_argument('--n_chunks', type=int, help=n_chunks_help)
parser.add_argument('--start', type=int, help=start_help)
parser.add_argument('--end', type=int, help=end_help)
parser.add_argument('--expname', type=str, help=expname_help)
parser.add_argument('--model', type=str, help=model_help)
parser.add_argument('--params', type=str, help=params_help)
parser.add_argument('--test', action='store_true', help=params_help)
parser.add_argument('--reset_period', type=int, default=-1, help=reset_help)
args = parser.parse_args()

if args.test:
  args.expname = 'test'
  args.model = 'inputs/model_data/marketscan.csv'
  args.params = 'test_data/test_params.csv'

# Create the save directory
util.Config.get_save_dir(args.expname)

param_dict = util.parse_parameters(args.params)

# list - dummy to create copies for different seeds
n_trials_list = np.arange(param_dict['n_trials'])


environment_name = param_dict['environment_name']

# Check environment and get additional keys for that env
environment_name = param_dict['environment_name']
environment_class = environments.Config.get_env_class(environment_name)
env_more_params = environment_class.environment_params().keys()

env_param_lists = [param_dict[key] for key in env_more_params]


all_param_combos = list(
    itertools.product(
        param_dict['n_arms_list'],
        param_dict['budget_frac_list'],
        param_dict['horizon_list'],
        param_dict['policy_list'],
        n_trials_list,
        *env_param_lists,
    )
)

num_unique_combos = len(all_param_combos)
if args.test:
  args.start = 0
  args.end = num_unique_combos

# print commands needed to run
if args.n_chunks:
  command_template = 'python run_experiments.py --start %s --end %s '
  command_template += '--expname %s --model %s --params %s --reset_period %s &'
  step_size = np.ceil(num_unique_combos/args.n_chunks)
  step_size = max(step_size, 1)
  chunks = np.arange(0, num_unique_combos, step_size).astype(int)
  n_chunks = chunks.shape[0]
  for chunk in range(n_chunks-1):
    index_start = chunks[chunk]
    index_end = chunks[chunk+1]
    str_data = (index_start, index_end, args.expname, args.model, args.params,
                args.reset_period)
    command_out = command_template % str_data
    print(command_out)

  # print the last chunk
  index_start = chunks[-1]
  index_end = num_unique_combos
  str_data = (index_start, index_end, args.expname, args.model, args.params,
              args.reset_period)
  command_out = command_template % str_data
  print(command_out)

  exit()

for experiment_index in range(args.start, args.end):
  combo = all_param_combos[experiment_index]
  n_arms = combo[0]
  budget_frac = combo[1]
  horizon = combo[2]
  policy = combo[3]
  trial_number = combo[4]  # placeholder to create unique seeds for given combo

  # make a parameter dict out of the remaining
  env_param_dict = collections.OrderedDict()
  for i, key in enumerate(env_more_params):
    env_param_dict[key] = combo[5+i]

  print(combo)

  # seed needs to be unique
  seed = experiment_index + param_dict['base_seed']

  # save meta data only on first experiment
  save_meta = trial_number == 0

  run_simulation(
      n_arms,
      budget_frac,
      param_dict['environment_name'],
      seed,
      trial_number,
      param_dict['base_seed'],
      horizon,
      policy,
      args.expname,
      save_meta=save_meta,
      model_data_file=args.model,
      env_param_dict=env_param_dict,
      stream_map=param_dict['stream_map'],
      reset_period=args.reset_period,
  )
