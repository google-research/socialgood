"""For simulating population disease progression under intervention policies.

Author: killian-34/jakillian
"""
import pickle
import time

import environments
import numpy as np
import policies
import util


def run_simulation(
    n_arms,
    budget_frac,
    environment_name,
    seed,
    trial_number,
    base_seed,
    horizon,
    policy_name,
    expname,
    env_param_dict=None,
    save_meta=False,
    model_data_file='',
    stream_map=None,
    reset_period=-1,
):
  """Run one simulation seed, given an environment, policy, and parameters."""
  # define instance data
  budget = int(n_arms*budget_frac)  # per-round intervention budget

  if stream_map:
    stream_map = util.get_stream_mappings_randomized(
        stream_map, n_arms, horizon
    )

  # set seed for reproducibility
  np.random.seed(seed)

  # during simulation, store all tuples of
  # (states, actions, rewards, next_states) for each policy
  data = {
      'states': np.zeros((n_arms, horizon), dtype=int),
      'actions': np.zeros((n_arms, horizon), dtype=int),
      'rewards': np.zeros((n_arms, horizon), dtype=float),
      'next_states': np.zeros((n_arms, horizon), dtype=int),
  }

  # Run the simulations
  start_seed = 0

  # Create environment
  start = time.time()
  environment_cls = environments.Config.get_env_class(environment_name)
  environment = environment_cls(
      n_arms, start_seed, horizon, model_data_file, stream_map, **env_param_dict
  )
  end = time.time()
  print('Done creating environment, took %s seconds' % (end - start))

  # Get file save name, unique to parameters of this experiment and seed
  filename = util.Config.get_exp_filename(
      environment,
      expname,
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

  print('policy %s' % policy_name)
  print('trial %s' % seed)

  # set seed for reproducibility between policies
  action_random_seed = seed
  np.random.seed(action_random_seed)

  # Initialize simulation environment with different seed
  add_to_env_seed = 1
  environment_random_seed = seed + add_to_env_seed
  environment.set_seed(environment_random_seed)
  environment.reset_states()

  # Get parameters to instantiate the policy class
  simulation_parameters = {}
  simulation_parameters['environment_info'] = environment.get_info()
  simulation_parameters['policy_parameters'] = {
      'index_lb': -1,
      'index_ub': 1,
      'gamma': 0.9999,
      'binary_search_tolerance': 0.01,
      'horizon': horizon,
      'lambda_lb': -1,
      'regularized_version': False,
      'n_robust_samples': 10,
      **simulation_parameters['environment_info'],
  }

  environment_rewards = environment.get_reward_definition()

  # Instantiate the policy class
  policy_cls = policies.Config.get_policy_class(policy_name)
  policy = policy_cls(
      n_arms=n_arms,
      budget=budget,
      transition_probs=environment.transition_probs,
      rewards=environment_rewards,
      **simulation_parameters['policy_parameters'])

  # Run the simulation
  states = environment.get_states()

  for t in range(horizon):
    print('timestep', t)
    # get data for one simluation step, and evolve environment
    remaining_horizon = horizon - t
    start = time.time()
    # state_tups = [environment.state_index_to_tuple[s] for s in states]
    actions = policy.get_actions(
        current_states=states, remaining_horizon=remaining_horizon)
    end = time.time()
    print('Getting actions took %s seconds' % (end - start))
    next_states, rewards = environment.step(actions)

    # log the data
    data['states'][:, t] = states
    data['actions'][:, t] = actions
    data['rewards'][:, t] = rewards
    data['next_states'][:, t] = next_states

    if reset_period > 0 and t > 0 and t%reset_period==0:
      environment.reset_states()
      policy.signal_reset()

    states = next_states

  # store results
  with open(results_filename, 'wb') as fo:
    pickle.dump(data, fo)

  if save_meta:
    with open(meta_filename, 'wb') as fo:
      simulation_parameters['environment_parameters'] = {
          'n_arms': n_arms,
          'budget': budget,
          'seed': seed,
          'horizon': horizon,
          'environment_name': environment_name,
      }
      pickle.dump(simulation_parameters, fo)

  # Print results for single run.
  print('-------------------')
  print('Simulation summary:')
  print('-------------------')
  print('n_arms: %i' % n_arms)
  print('budget: %i' % budget)
  print('horizon length: %i' % horizon)
  print('seed: %i' % seed)

  window = 10
  print('window size:',window)
  for key in env_param_dict.keys():
    print('%s: %s'%(key, env_param_dict[key]))
  print()
  print('Policy: %s' % policy_name)
  cumulative_reward = data['rewards'][:, -window:].sum(axis=-1)
  mean_cumulative_reward = cumulative_reward.mean() / window
  print('Mean Cumulative Reward, all arms: %.2f' %
        (mean_cumulative_reward))
  print()
  group_map = environment.group_map
  for group in set(environment.group_map):
    group_inds = group_map == group
    mean_cumulative_reward = cumulative_reward[group_inds].mean() / window
    print('Mean Cumulative Reward, group %s: %.2f' %
          (group, mean_cumulative_reward))
    print()

