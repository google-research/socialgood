"""Utility functions for the repo, especially file parsing."""

import abc  # create abstract base classes
import os

import environments
from numba import jit
import numpy as np


class Config():
  """Static Config class for handling file naming and saving."""

  DATA_DIRECTORY = 'data'
  IMG_DIRECTORY = 'outputs/img'
  CSV_SUMMARY_DIRECTORY = 'outputs/csv_summaries'
  EXP_FILENAME_TEMPLATE = (
      '%s_narms-%s_budgetfrac-%.2f_env-%s_trial-%i_bseed-%s_h-%i_policy-%s'
  )
  COMBINED_FILENAME_TEMPLATE = (
      '%s_COMBINED_narms-%s_budgetfrac-%.2f_env-%s_trials-%i_bseed-%i_h-%i'
  )
  BASE_RESULTS_PATH = os.path.join(DATA_DIRECTORY, 'results')
  BASE_LP_MODEL_PATH = os.path.join(DATA_DIRECTORY, 'base_lp_models')

  FORMATTER = {
      'float': '%.2f',
      'int': '%i'
  }

  @staticmethod
  def get_img_output_path(exp_dir_name):
    """Get img output filepath."""
    img_dir_path = os.path.join(Config.IMG_DIRECTORY, exp_dir_name)
    if not os.path.exists(img_dir_path):
      os.makedirs(img_dir_path)
    return img_dir_path

  @staticmethod
  def get_csv_summary_output_path(exp_dir_name):
    """Get csv output summary filepath."""
    csv_summary_dir_path = os.path.join(
        Config.CSV_SUMMARY_DIRECTORY, exp_dir_name
    )
    if not os.path.exists(csv_summary_dir_path):
      os.makedirs(csv_summary_dir_path)
    return csv_summary_dir_path

  @staticmethod
  def get_filename_template(template, env_param_types):
    """Get filename template, given env parameters."""
    filename = template
    for param in env_param_types.keys():
      param_type = env_param_types[param]
      param_formatter = Config.FORMATTER[param_type]
      new_tag = '_' + param  + '-' + param_formatter
      filename += new_tag
    return filename

  @staticmethod
  def get_save_dir(exp_dir_name):
    """Get save directory."""
    save_dir = os.path.join(Config.BASE_RESULTS_PATH, exp_dir_name)
    if not os.path.exists(save_dir):
      os.mkdir(save_dir)
    return save_dir

  @staticmethod
  def get_exp_filename(
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
  ):
    """Get experiment filename."""
    save_dir = Config.get_save_dir(expname)

    env_param_types = environment.environment_params()

    # Create filename with unique param combo of this run,
    # including specific env params
    filename_template = Config.get_filename_template(
        Config.EXP_FILENAME_TEMPLATE, env_param_types
    )
    filename = os.path.join(save_dir, filename_template)

    format_values = (expname, n_arms, budget_frac, environment_name)
    format_values += (trial_number, base_seed, horizon, policy_name)
    format_values += tuple(env_param_dict.values())

    filename = filename%format_values

    return filename

  @staticmethod
  def get_combined_filename(
      environment,
      expname,
      n_arms,
      budget_frac,
      environment_name,
      n_trials,
      base_seed,
      horizon,
      env_param_dict,
  ):
    """Get combined filename."""
    filename = Config.get_combined_filename_template(environment, expname)

    format_values = (expname, n_arms, budget_frac, environment_name)
    format_values += (n_trials, base_seed, horizon)
    format_values += tuple(env_param_dict.values())

    filename = filename%format_values

    return filename

  @staticmethod
  def get_combined_filename_template(environment, expname):
    """Get combined filename template."""
    save_dir = Config.get_save_dir(expname)

    env_param_types = environment.environment_params()

    # Add specific env params
    filename_template = Config.get_filename_template(
        Config.COMBINED_FILENAME_TEMPLATE, env_param_types
    )
    filename = os.path.join(save_dir, filename_template)

    return filename


@jit(nopython=True)
def finite_horizon_value_iteration(timesteps, p, r, act_state, discount=1.0):
  """Finite horizon value iteration.

  Note: This implementation gives a 50x speedup over mdptoolbox's
  FiniteHorizon.run() solution method.
  Note that numba is fastest when the code is written
  with unrolled for loops, due to how numba compiles the
  function behind the scenes. This is faster than
  vectorized numpy operations.

  Args:
    timesteps: number of timesteps to compute value function
    p: 3-d array of transition probabilities in AxSxS format
    r: 3-d array of rewards in AxSxS format
    act_state: state of the value function to return
    discount: discount factor in [0, 1]

  Returns:
    1: value function of act_state
    2: and best action to take in act_state
  """
  n_actions = p.shape[0]
  n_states = p.shape[1]
  q = np.zeros((n_actions, n_states), dtype=np.float64)
  v = np.zeros((n_states, timesteps+1), dtype=np.float64)

  v_max = np.zeros(n_states, dtype=np.float64)

  for t in range(timesteps):
    v_sub = v[:, timesteps - t]
    for aa in range(n_actions):
      for s in range(n_states):
        expected = 0
        for sp in range(n_states):
          expected += p[aa, s, sp] * v_sub[sp]
        q[aa, s] = r[aa, s] + discount * expected

    stage = timesteps - t - 1
    for state in range(n_states):
      v_max[state] = np.max(q[:, state])
    v[:, stage] = v_max

  best_action = np.argmax(q[:, act_state])

  return v[act_state, 0], best_action


def parse_parameters(filename):
  """Parse input experiment parameter file."""
  expected_keys = {
      'budget_frac_list': 'float_list',
      'n_arms_list': 'int_list',
      'horizon_list': 'int_list',
      'policy_list': 'str_list',
      'environment_name': 'str',
      'base_seed': 'int',
      'n_trials': 'int',
      'stream_map': 'float_list'
  }

  all_expected_keys = set(expected_keys.keys())
  params = {key: None for key in all_expected_keys}

  # parse file, check for all keys
  handle = open(filename, 'r')
  _ = handle.readline()  # strip header

  keys_in = set()
  for line in handle:
    param_line = line.strip().split(',')
    param_name = param_line[0]
    keys_in.add(param_name)

    # keep list, ended by blank entry or end of line
    index_end = len(param_line)
    if '' in param_line:
      index_end = param_line.index('')
    params[param_name] = param_line[1:index_end]

  # raise error if there is a missing expected key
  missing_keys = all_expected_keys - keys_in
  if missing_keys:
    err_string = 'Expected these keys in parameter file "%s", '
    err_string += 'but were missing: %s'
    raise ValueError(err_string%(filename, missing_keys))

  # Check environment and get additional keys for that env
  environment_name = params['environment_name'][0]
  environment_class = environments.Config.get_env_class(environment_name)
  env_more_keys = environment_class.environment_params()

  for key in env_more_keys.keys():
    expected_keys[key] = env_more_keys[key]+'_list'

  # raise error if there is a missing expected key for the environment
  missing_keys = set(env_more_keys.keys()) - keys_in
  if missing_keys:
    err_string = 'Expected these keys in parameter file "%s" for env'
    err_string += ' "%s" but were missing: %s'
    raise KeyError(err_string%(filename, environment_name, missing_keys))

  # Convert strs
  check_list = [
      key for key in expected_keys.keys() if expected_keys[key] == 'str'
  ]
  for key in check_list:
    params[key] = params[key][0]

  # Convert ints
  check_list = [
      key for key in expected_keys.keys() if expected_keys[key] == 'int'
  ]
  for key in check_list:
    params[key] = int(params[key][0])

  # Convert int lists
  check_list = [
      key for key in expected_keys.keys() if expected_keys[key] == 'int_list'
  ]
  for key in check_list:
    params[key] = list(map(int, params[key]))

  # Convert float lists
  check_list = [
      key for key in expected_keys.keys() if expected_keys[key] == 'float_list'
  ]
  for key in check_list:
    params[key] = list(map(float, params[key]))

  # Finally, delete any unexpected keys
  unexpected_keys = keys_in - set(expected_keys.keys())
  for key in unexpected_keys:
    del params[key]

  return params


def get_stream_mappings_deterministic(start_at_i_fracs, n_arms, horizon):
  """Get stream mapping, with deterministic of arms order from start."""
  # start_at_i_fracs must sum to 1
  eps = 1e-6
  assert abs(sum(start_at_i_fracs) - 1) <= eps

  # If map is longer than horizon, truncate
  if len(start_at_i_fracs) > horizon:
    start_at_i_fracs_truncated = start_at_i_fracs[:horizon]
    start_at_i_fracs_truncated[horizon-1] = sum(start_at_i_fracs[horizon-1:])
    start_at_i_fracs = start_at_i_fracs_truncated

  stream_map = np.zeros(n_arms)
  previous_index = 0
  for i, frac in enumerate(start_at_i_fracs):
    next_index = int(n_arms*frac) + previous_index
    stream_map[previous_index:next_index] = i
    previous_index = next_index

  return stream_map


def get_stream_mappings_randomized(start_at_i_fracs, n_arms, horizon):
  """Get stream mapping, randomzing which arms start when."""
  # start_at_i_fracs must sum to 1
  eps = 1e-6
  assert abs(sum(start_at_i_fracs) - 1) <= eps

  # If map is longer than horizon, truncate
  if len(start_at_i_fracs) > horizon:
    start_at_i_fracs_truncated = start_at_i_fracs[:horizon]
    start_at_i_fracs_truncated[horizon-1] = sum(start_at_i_fracs[horizon-1:])
    start_at_i_fracs = start_at_i_fracs_truncated

  stream_map = np.zeros(n_arms)
  arms_not_assigned = set(np.arange(n_arms))
  for i, frac in enumerate(start_at_i_fracs):
    num_arms_to_assign = int(frac * n_arms)
    arms_starting_at_i = np.random.choice(
        list(arms_not_assigned), size=num_arms_to_assign, replace=False
    )
    stream_map[arms_starting_at_i] = i
    arms_not_assigned -= set(arms_starting_at_i)

  return stream_map


class GetActiveArms(abc.ABC):
  """Abstract class for GetActiveArm helper class.

    Want to be able to define get_active_arms functions
    within environments. But we also want policy functions
    to be able to call it and we want to be able to save the policy class
    instances without saving the environment classes, since they can be very
    large. so this class is the go-between.
  """

  @abc.abstractmethod
  def get_active_arms(self, states: np.ndarray = None) -> np.ndarray:
    """Return the active arms of an RMABEnvironment."""
    pass


class GetActiveArmsDiabetesApp(GetActiveArms):
  """Class for GetActiveArms helper class for diabetes app environment."""

  def __init__(self, state_index_to_tuple, active_code, active_index) -> None:
    """Initialize the GetActiveArmsDiabetesApp class."""

    self.state_index_to_tuple = state_index_to_tuple
    self.active_code = active_code
    self.active_index = active_index

  def get_active_arms(self, states):
    """Initialize the GetActiveArmsDiabetesApp class."""

    active_arms = np.zeros(states.shape[0], dtype=int)
    for i, s in enumerate(states):
      state_tup = self.state_index_to_tuple[s]
      s_c = state_tup[self.active_index]
      if s_c == self.active_code:
        active_arms[i] = 1

    return active_arms


class GetActiveArmsDefault(GetActiveArms):
  """Class for GetActiveArms helper class for diabetes app environment."""

  def __init__(self) -> None:
    """Initialize the GetActiveArmsDefault class."""

  def get_active_arms(self, states):
    """Return array of ones, indicating all arms are active."""
    return np.ones(states.shape[0], dtype=int)

