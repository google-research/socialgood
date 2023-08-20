"""Defines environments for RMAB disease simulations.

Contains classes that define simulation environments,
e.g., for joint app engagement and diabetes disease progression.

Author: killian-34/jakillian
"""

import abc  # create abstract base classes
import collections
import functools
import itertools
import math

import numpy as np
import pandas as pd
import util


class RMABEnvironment(abc.ABC):
  """Abstract base class for RMAB environments."""

  @abc.abstractmethod
  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               ) -> None:
    """Initialize the abstract RMABEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to instantiate a model
        e.g., probabilities, rewards, and group maps
      stream_map: defines when each arm enters the simulation
    """
    self.n_arms = n_arms
    self.start_seed = start_seed
    self.horizon = horizon
    self.model_data_file = model_data_file
    self.stream_map = stream_map

    # create random stream and set seed
    self.random_stream = np.random.RandomState()
    self.set_seed(self.start_seed)

    self.current_states = None
    self.transition_probs = None
    self.rewards = None

    self.group_map = None

  def set_seed(self, seed: int) -> None:
    """Set the random seed."""
    self.random_stream.seed(seed)

  def step(self, actions: np.ndarray) -> np.ndarray:
    """Evolve all arm states by a single step.

    Args:
      actions: n_arms-length array of actions, encoded as integers

    Returns:
      next_states: n_arms-length array of arm states after being evolved
      rewards: n_arms-length array of rewards

    """
    next_states = np.zeros(self.n_arms, dtype=int)
    for n in range(self.n_arms):
      state_ind = self.current_states[n]
      next_state_one_hot = self.random_stream.multinomial(
          n=1,
          pvals=self.transition_probs[n, state_ind, actions[n]],
      )
      next_state_ind = np.argmax(next_state_one_hot)
      next_states[n] = next_state_ind

    rewards = self.get_rewards_for_states(self.current_states)
    self.current_states = np.copy(next_states)

    return next_states, rewards

  def get_rewards_for_states(self, current_states: np.ndarray) -> np.ndarray:
    """Return rewards for the current arm states.

    Args:
      current_states: n_arms-length array of integer arm states

    Returns:
      rewards: reward value for each arm, based on its state

    """
    return np.array(
        [self.rewards[n, current_states[n]] for n in range(self.n_arms)])

  @staticmethod
  def parse_model_data(filename, n_arms):
    """Default parser for RMABEnvironments, only expecting group definitions."""
    df = pd.read_csv(filename)
    expected_cols = ['Group', 'frac']

    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_mappings = np.zeros(n_arms, dtype=np.int32)
    prev_mapping_ind = 0
    for row in range(df.shape[0]):

      group = df.iloc[row]['Group']

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

    return group_mappings

  def get_reward_definition(self) -> np.ndarray:
    """Return a copy of the reward definition."""
    return np.copy(self.rewards)

  def get_states(self) -> np.ndarray:
    """Return the current arm states."""
    return np.copy(self.current_states)

  def get_info(self) -> dict[str, object]:
    """Return other relevant environment information."""
    return {
        'group_map': self.group_map,
        'active_arms_helper': self.create_active_arms_helper(),
    }

  def create_active_arms_helper(self) -> util.GetActiveArms:
    self.active_arms_helper = util.GetActiveArmsDefault()
    return self.active_arms_helper

  def get_active_arms(self, states):
    """Method to get active arms of this environment."""
    return self.active_arms_helper.get_active_arms(states)

  @abc.abstractmethod
  def reset_states(self) -> np.ndarray:
    """Set all arm states to some initial value, and return the states."""
    pass

  @abc.abstractmethod
  def get_transition_probabilities(self) -> None:
    """Set the arm transition probabilities."""
    pass

  @abc.abstractmethod
  def set_reward_definition(self) -> None:
    """Set all arm reward definitions."""
    pass

  @staticmethod
  @abc.abstractmethod
  def env_key() -> str:
    """Return the name of the environment."""
    raise NotImplementedError

  @staticmethod
  @abc.abstractmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    raise NotImplementedError


class RandomEnvironment(RMABEnvironment):
  """Class for generating uniform random RMAB environments."""

  env_key_name = 'Random Environment'

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               n_states: int = 2,
               # n_actions: int = 2
               ) -> None:
    """Initialize the RandomEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_map
      stream_map: unused by this class
      n_states: number of states per arm
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    group_map = RandomEnvironment.parse_model_data(model_data_file, n_arms)
    self.group_map = group_map

    self.n_states = n_states
    self.n_actions = 2  # make parameter if there are multi-action policeies

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    self.current_states = np.zeros(n_arms, dtype=np.int32)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((n_arms, n_states), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return RandomEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    params['n_states'] = 'int'
    return params

  def get_transition_probabilities(self) -> None:
    """Samples a transition function for each arm in a uniform random manner."""
    for n in range(self.n_arms):
      for s in range(self.n_states):
        for a in range(self.n_actions):
          self.transition_probs[n, s, a] = self.random_stream.dirichlet(
              alpha=np.ones(self.n_states))

  def reset_states(self) -> np.ndarray:
    """Reset the state of all arms in a uniform random manner."""
    self.current_states = self.random_stream.choice(
        a=np.arange(self.n_states), size=self.n_arms, replace=True)
    return self.current_states

  def set_reward_definition(self) -> None:
    """Set the reward definition to identity."""
    self.rewards[:] = np.arange(self.n_states)


class TwoStateEnvironment(RMABEnvironment):
  """Class for generating 2-state environments."""

  env_key_name = 'Two State Environment'

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               ) -> None:
    """Initialize the TwoStateEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_map
      stream_map: unused by this class
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    ret_data = TwoStateEnvironment.parse_model_data(
        model_data_file, 
        n_arms
    )
    self.group_probs = ret_data[0]
    self.group_map = ret_data[1]
    self.group_rewards = ret_data[2]

    self.n_states = 2
    self.n_actions = 2

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    self.current_states = np.zeros(n_arms, dtype=np.int32)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((n_arms, self.n_states), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return TwoStateEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    return params


  @staticmethod
  def parse_model_data(filename, n_arms):
    """Parse diabetes model data."""
    df = pd.read_csv(filename)
    prob_cols = [
        'p_s0_a0_s1',
        'p_s1_a0_s1',
        'p_s0_a1_s1',
        'p_s1_a1_s1',
    ]
    reward_cols = [
        'r_s0',
        'r_s1',
    ]
    expected_cols = ['Group', 'frac'] + reward_cols + prob_cols

    # print(df['frac'].sum())
    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_probs = [{} for group in range(df.shape[0])]
    group_rewards = [{} for group in range(df.shape[0])]
    group_mappings = np.zeros(n_arms, dtype=np.int32)
    group_definitions = []
    prev_mapping_ind = 0
    for row in range(df.shape[0]):
      group = int(df.iloc[row]['Group'])

      # assign probability parameters
      for param in prob_cols:
        group_probs[group][param] = df.iloc[row][param]

      # assign reward parameters
      for param in reward_cols:
        group_rewards[group][param] = df.iloc[row][param]

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

    return group_probs, group_mappings, group_rewards


  def get_transition_probabilities(self) -> None:
    """Load parameters from dict."""
    for n in range(self.n_arms):
      group = self.group_map[n]
      for s in range(self.n_states):
        for a in range(self.n_actions):
          prob_str = 'p_s%i_a%i_s1'%(s, a)
          self.transition_probs[n, s, a, 1] = self.group_probs[group][prob_str]
          remaining = 1 - self.transition_probs[n, s, a, 1]
          self.transition_probs[n, s, a, 0] = remaining

  def reset_states(self) -> np.ndarray:
    """Reset the state of all arms in a uniform random manner."""
    self.current_states = self.random_stream.choice(
        a=np.arange(self.n_states), size=self.n_arms, replace=True)
    return self.current_states

  def set_reward_definition(self) -> None:
    """Set the reward definition."""
    for n in range(self.n_arms):
      group = self.group_map[n]
      for s in range(self.n_states):
        reward_str = 'r_s%i' % s
        self.rewards[n, s] = self.group_rewards[group][reward_str]


class MaternalHealthEnvironment(RMABEnvironment):
  """Class for generating 3-state maternal health environment."""

  env_key_name = 'Maternal Health Environment'

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               ) -> None:
    """Initialize the MaternalHealthEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_map
      stream_map: unused by this class
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    ret_data = MaternalHealthEnvironment.parse_model_data(
        model_data_file, 
        n_arms
    )
    self.group_probs = ret_data[0]
    self.group_map = ret_data[1]
    self.group_rewards = ret_data[2]

    self.n_states = 3
    self.n_actions = 2

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    self.current_states = np.zeros(n_arms, dtype=np.int32)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((n_arms, self.n_states), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return MaternalHealthEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    return params


  @staticmethod
  def parse_model_data(filename, n_arms):
    """Parse maternal health model data."""
    df = pd.read_csv(filename)
    prob_cols = [
        'p000',
        'p010',
        'p102',
        'p110',
        'p202',
        'p212',
    ]
    reward_cols = [
        'r0',
        'r1',
        'r2',
    ]
    expected_cols = ['Group', 'frac'] + reward_cols + prob_cols

    # print(df['frac'].sum())
    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_probs = [{} for group in range(df.shape[0])]
    group_rewards = [{} for group in range(df.shape[0])]
    group_mappings = np.zeros(n_arms, dtype=np.int32)
    group_definitions = []
    prev_mapping_ind = 0
    for row in range(df.shape[0]):
      group = int(df.iloc[row]['Group'])

      # assign probability parameters
      for param in prob_cols:
        group_probs[group][param] = df.iloc[row][param]

      # assign reward parameters
      for param in reward_cols:
        group_rewards[group][param] = df.iloc[row][param]

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

    return group_probs, group_mappings, group_rewards

  def sample_prob(self, prob):
    """Sample transition probabilities with some noise."""
    tiny_eps = 1e-3
    std = 0.2

    # group = self.group_map[arm]
    # prob = self.group_probs[group][prob_name]

    # want to scale deviations based on their distance from 0 or 1
    scale = min(abs(prob), abs(1-prob))

    # introduce small variation between arms in each group
    sigma = std*scale
    sampled_prob = min(1-tiny_eps, self.random_stream.normal(prob, sigma))
    sampled_prob = max(tiny_eps, sampled_prob)
    # self.arm_probs[arm][prob_name] = sampled_prob

    return sampled_prob


  def get_transition_probabilities(self) -> None:
    """Load parameters from dict."""

    for n in range(self.n_arms):

      group = self.group_map[n]
      
      tup_list = [
        (0, 0, 0),
        (0, 1, 0),
        (1, 0, 2),
        (1, 1, 0),
        (2, 0, 2),
        (2, 1, 2),
      ]
      for tup in tup_list:
        s, a, sp = tup
        prob_str = 'p%i%i%i'%(s, a, sp)
        prob_mean = self.group_probs[group][prob_str]
        prob = self.sample_prob(prob_mean)
        self.transition_probs[n, s, a, sp] = prob
        self.transition_probs[n, s, a, 1] = 1 - prob


  def reset_states(self) -> np.ndarray:
    """Reset the state of all arms in a uniform random manner."""
    self.current_states = self.random_stream.choice(
        a=np.arange(self.n_states), size=self.n_arms, replace=True)
    return self.current_states

  def set_reward_definition(self) -> None:
    """Set the reward definition."""
    for n in range(self.n_arms):
      group = self.group_map[n]
      for s in range(self.n_states):
        reward_str = 'r%i' % s
        self.rewards[n, s] = self.group_rewards[group][reward_str]


class DiabetesAppEnvironment(RMABEnvironment):
  """Class for the Diabetes App Environment."""

  env_key_name = 'Diabetes App Environment'

  DROPOUT = 0
  MAINTENANCE = 1
  ENGAGED = 2

  A1CGT8 = 0
  A1CLT8 = 1

  # Value of s_counter when arm becomes active (streaming)
  ARM_ACTIVE = 0

  # index position of each state within the state tuple
  HEALTH_INDEX = 1
  ACTIVE_INDEX = 3

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               alpha: float = 0.5,
               # group_map: np.ndarray = None,
               # group_probs: list[dict[str, float]] = None,
               # group_rewards: list[dict[str, float]] = None
               ) -> None:
    """Initialize the DiabetesAppEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_probs, group_map,
        and group_rewards
      stream_map: defines when each arm enters the simulation.
      alpha: weight to place on engagement rewards (remaining weight goes on
        health rewards)
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    data = DiabetesAppEnvironment.parse_model_data(model_data_file, n_arms)
    self.group_probs = data[0]
    self.group_map = data[1]
    self.group_definitions = data[2]
    self.group_rewards = data[3]

    self.n_actions = 2
    self.alpha = alpha
    self.horizon = horizon
    self.stream_map = stream_map

    self.stream_horizon = 12

    # if self.group_map is None:
    #   self.group_map = np.zeros(self.n_arms, dtype=np.int32)
    # else:
    #   self.group_map = group_map

    # self.group_probs = group_probs

    # if no streaming is defined, all arms start on round 0
    if self.stream_map is None:
      self.stream_map = np.zeros(self.n_arms)

    # if no group rewards defined, use these defaults
    # self.group_rewards = group_rewards
    if self.group_rewards is None:
      self.group_rewards = []
      for group in set(self.group_map):
        self.group_rewards[group] = {
            'r_dropout': 0,
            'r_maintenance': 1,
            'r_eng': 1,
            'r_a1cgt8': 0,
            'r_a1clt8': 1,
        }

    # Note: MAINTENANCE and DROPOUT get treated the same in
    # memory state, so represent both as 1 and have only 4 memory states
    self.state_dimension_counts = [
        3,  # Engagement dimension: dropout=0, maint=1, engaged=2
        2,  # HbA1c dimension: 'A1c >= 8'=0, 'A1c < 8'=1
        4,  # Memory dimension: ('m', 'm')=(1, 1), ('m', 'e')=(1, 2), ...
    ]

    self.engagement_states = np.arange(self.state_dimension_counts[0])
    self.health_states = np.arange(self.state_dimension_counts[1])
    self.memory_states = list(itertools.product([1, 2], repeat=2))

    # streaming states will be a linear addition to the state space count
    self.streaming_states = np.arange(self.stream_horizon)

    self.n_states = math.prod(self.state_dimension_counts)
    self.n_states += len(self.streaming_states)-1

    # create the combined state reference dictionaries
    ret_data = self.get_state_dicts_combined()
    self.state_tuple_to_index = ret_data[0]
    self.state_index_to_tuple = ret_data[1]

    # create the state reference dictionaries for the memory dimension
    ret_data = self.get_state_dicts_memory()
    self.state_tuple_to_index_mem = ret_data[0]
    self.state_index_to_tuple_mem = ret_data[1]

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    # design choice: states will be kept and returned as indexes
    self.current_states = np.zeros(self.n_arms, dtype=int)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((self.n_arms, self.n_states), dtype=np.float)
    self.rewards_2d = np.zeros((self.n_arms, self.n_states, 2), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return DiabetesAppEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    params['alpha'] = 'float'
    return params

  @staticmethod
  def parse_model_data(filename, n_arms):
    """Parse diabetes model data."""
    df = pd.read_csv(filename)
    prob_cols = [
        'p_i_mtoe',
        'p_i_mtod',
        'p_i_etoe',
        'p_u_mtod',
        'p_noeng_gl',
        'p_noeng_ll',
        'p_eng_gl',
        'p_eng_ll',
    ]
    reward_cols = [
        'r_dropout',
        'r_maintenance',
        'r_eng',
        'r_a1cgt8',
        'r_a1clt8',
    ]
    expected_cols = ['Group', 'sex', 'age', 'frac'] + reward_cols + prob_cols

    # print(df['frac'].sum())
    assert not set(expected_cols) - set(df.columns.values)
    assert abs(df['frac'].sum() - 1) < 1e-4

    group_probs = [{} for group in range(df.shape[0])]
    group_rewards = [{} for group in range(df.shape[0])]
    group_mappings = np.zeros(n_arms, dtype=np.int32)
    group_definitions = []
    prev_mapping_ind = 0
    for row in range(df.shape[0]):
      group = df.iloc[row]['Group']

      # assign probability parameters
      for param in prob_cols:
        group_probs[group][param] = df.iloc[row][param]

      # assign reward parameters
      for param in reward_cols:
        group_rewards[group][param] = df.iloc[row][param]

      # build group mappings
      frac = df.iloc[row]['frac']
      next_mapping_ind = int(n_arms*frac)
      group_mappings[prev_mapping_ind:prev_mapping_ind+next_mapping_ind] = group
      prev_mapping_ind += next_mapping_ind

      # build group to descriptor map
      tup = tuple(df.iloc[row][['sex', 'age']].values)
      group_definitions.append(tup)

    return group_probs, group_mappings, group_definitions, group_rewards

  def create_active_arms_helper(self):
    self.active_arms_helper = util.GetActiveArmsDiabetesApp(
        self.state_index_to_tuple,
        self.ARM_ACTIVE,
        self.ACTIVE_INDEX,
    )
    return self.active_arms_helper

  # def get_active_arms(self, states):
  #   """Method to get active arms of this environment."""
  #   return self.active_arms_helper.get_active_arms(states)
  #   # active_arms = np.zeros(states.shape[0], dtype=int)
  #   # for i, s in enumerate(states):
  #   #   state_tup = self.state_index_to_tuple[s]
  #   #   s_c = state_tup[5]
  #   #   if s_c == 0:
  #   #     active_arms[i] = 1
  #   # return active_arms

  def get_info(self) -> dict[str, object]:
    """Return the state index-to-tuple translators."""
    return {
        'state_index_to_tuple': self.state_index_to_tuple,
        'state_index_to_tuple_mem': self.state_index_to_tuple_mem,
        'state_tuple_to_index': self.state_tuple_to_index,
        'state_tuple_to_index_mem': self.state_tuple_to_index_mem,
        'rewards_2d': self.rewards_2d,
        'alpha': self.alpha,
        'high_a1c_code': self.A1CGT8,
        'state_tuple_health_index': self.HEALTH_INDEX,
        'group_map': self.group_map,
        'active_arms_helper': self.create_active_arms_helper(),
    }

  def get_state_dicts_memory(
      self
  ) -> (dict[tuple[int, int, tuple[int, int]], int], dict[
      int, [tuple[int, int, tuple[int, int]]]]):
    """Create the state reference dictionaries for the memory dimension."""

    index_to_tuple = dict(list(enumerate(self.memory_states)))
    tuple_to_index = dict([
        (value, key) for key, value in index_to_tuple.items()
    ])
    return tuple_to_index, index_to_tuple

  def get_state_dicts_combined(
      self
  ) -> (dict[tuple[int, int, tuple[int, int]], int], dict[int, tuple[
      int, int, tuple[int, int]]]):
    """Create the state reference dictionaries over all states."""
    all_active_tuples = list(
        itertools.product(
            self.engagement_states,
            self.health_states,
            self.memory_states,
            [self.ARM_ACTIVE],  # stream counter is 0 when arm is active
            repeat=1))

    s_e = self.engagement_states[0]
    s_h = self.health_states[0]
    s_m = self.memory_states[0]
    stream_tuples = [(s_e, s_h, s_m, s_c) for s_c in range(1, self.stream_horizon)]

    # add stream counter states to the end
    all_tuples = all_active_tuples + stream_tuples

    index_to_tuple = dict(list(enumerate(all_tuples)))
    tuple_to_index = dict([
        (value, key) for key, value in index_to_tuple.items()
    ])
    return tuple_to_index, index_to_tuple

  def sample_and_store(self, prob_name, arm):
    """Sample transition probabilities with some noise."""
    tiny_eps = 0.025
    std = 0.5

    group = self.group_map[arm]
    prob = self.group_probs[group][prob_name]

    # want to scale deviations based on their distance from 0 or 1
    scale = min(abs(prob), abs(1-prob))

    # introduce small variation between arms in each group
    sigma = std*scale
    sampled_prob = min(1-tiny_eps, self.random_stream.normal(prob, sigma))
    sampled_prob = max(tiny_eps, sampled_prob)
    self.arm_probs[arm][prob_name] = sampled_prob

    return sampled_prob

  def get_transition_probabilities(self) -> None:
    """Create the app-engagement/diabetes progression RMAB."""
    self.arm_probs = [{} for arm in range(self.n_arms)]

    for arm_i in range(self.n_arms):

      p_i_mtoe = self.sample_and_store('p_i_mtoe', arm_i)
      p_i_mtod = self.sample_and_store('p_i_mtod', arm_i)
      p_u_mtod = self.sample_and_store('p_u_mtod', arm_i)
      p_i_etoe = self.sample_and_store('p_i_etoe', arm_i)

      p_eng_gl = self.sample_and_store('p_eng_gl', arm_i)
      p_eng_ll = self.sample_and_store('p_eng_ll', arm_i)
      p_noeng_gl = self.sample_and_store('p_noeng_gl', arm_i)
      p_noeng_ll = self.sample_and_store('p_noeng_ll', arm_i)

      transition_probs_one_arm = np.zeros(
          (self.n_states, self.n_actions, self.n_states))

      # The workflow for defining transition probabilities is as follows:
      # 0. First, streaming states are "tacked on", since they are
      #    disjoint from the rest of the state space evolution.
      # 1. For each state dimension, create a vector of length equal
      #    to the size of that dimension's state space. This will hold
      #    the probabilities of transitioning to each of the states next round.
      # 2. Once we have a probability vector for each dimension, take the
      #    product of each to get the final transition row.
      # Note: This method does not allow for defining joint transition
      #    probabilities between state dimensions. To do that, we have to
      #    take a kroneker product of their vectors before
      #    defining the probabilities for the joint space. See
      #    DiabetesAppPartialObsEnvironment for an example of joint
      #    definitions work. Much more logic/code is required.

      # Step 0: first add streaming transition definitions
      # loop backwards from latest possible start
      def exit_streaming_start_distribution(prob_row):
        # for now, have all arms start in engaged, mem=[M, M] high A1c
        s_e = self.ENGAGED
        s_h = self.A1CGT8
        s_m = (self.MAINTENANCE, self.MAINTENANCE)
        s_c = self.ARM_ACTIVE
        state_tup = (s_e, s_h, s_m, s_c)
        index = self.state_tuple_to_index[state_tup]
        prob_row[index] = 1

      for s_c in range(self.stream_horizon-1, 0, -1):

        # setting up streaming counter states, define some default values
        s_e = self.engagement_states[0]
        s_h = self.health_states[0]
        s_m = self.memory_states[0]
        row_tup = (s_e, s_h, s_m, s_c)

        transition_row = np.zeros(self.n_states)

        # count down
        if s_c > 1:
          next_state_tup = (s_e, s_h, s_m, s_c-1)
          index = self.state_tuple_to_index[next_state_tup]
          transition_row[index] = 1
        elif s_c == 1:
          # Probability distro over start states after streaming in
          exit_streaming_start_distribution(transition_row)

        state_index = self.state_tuple_to_index[row_tup]
        transition_probs_one_arm[state_index, :] = transition_row

      # Steps 1 and 2: Define the transition probabilities for the
      # non-streaming states
      s_c_active = self.ARM_ACTIVE

      for s_e in self.engagement_states:
        for s_h in self.health_states:
          for s_m in self.memory_states:
            for a in range(self.n_actions):

              engagement_row = np.zeros(self.state_dimension_counts[0])
              health_row = np.zeros(self.state_dimension_counts[1])
              memory_row = np.zeros(self.state_dimension_counts[2])
              stream_row = np.zeros(self.stream_horizon - 1)

              # Step 1.1: Define memory dimension probabilities
              #   memory changes are deterministic, just slide previous states
              #   and ensure that 0s (DROPOUTS) become 1s in the memory tuple
              memory_tup_next = (max(1, s_e), s_m[0])
              memory_ind_next = self.state_tuple_to_index_mem[memory_tup_next]
              memory_row[memory_ind_next] = 1

              # Step 1.2: Define Engagement dimension probabilities.
              if s_e == self.ENGAGED:
                if a == 1:  # intervention
                  engagement_row[self.ENGAGED] = p_i_etoe
                  engagement_row[self.MAINTENANCE] = 1 - p_i_etoe

                elif a == 0:  # no intervention
                  engagement_row[self.MAINTENANCE] = 1

              elif s_e == self.MAINTENANCE:
                if a == 1:
                  engagement_row[self.ENGAGED] = p_i_mtoe
                  engagement_row[self.MAINTENANCE] = 1 - p_i_mtod - p_i_mtoe
                  engagement_row[self.DROPOUT] = p_i_mtod

                elif a == 0:
                  engagement_row[self.MAINTENANCE] = 1 - p_u_mtod
                  engagement_row[self.DROPOUT] = p_u_mtod

              elif s_e == self.DROPOUT:
                engagement_row[self.DROPOUT] = 1

              # Step 1.3: Define A1c dimension probabilities.
              if s_m[1] == self.ENGAGED:
                if s_h == self.A1CGT8:
                  health_row[self.A1CLT8] = p_eng_gl
                  health_row[self.A1CGT8] = 1 - p_eng_gl
                elif s_h == self.A1CLT8:
                  health_row[self.A1CLT8] = p_eng_ll
                  health_row[self.A1CGT8] = 1 - p_eng_ll
              elif s_m[1] != self.ENGAGED:
                if s_h == self.A1CGT8:
                  health_row[self.A1CLT8] = p_noeng_gl
                  health_row[self.A1CGT8] = 1 - p_noeng_gl
                elif s_h == self.A1CLT8:
                  health_row[self.A1CLT8] = p_noeng_ll
                  health_row[self.A1CGT8] = 1 - p_noeng_ll

              # Step 2: Take product of probabilities.
              # Product needs to be in this order to match the state definition.
              transition_row = functools.reduce(
                  np.kron, [engagement_row, health_row, memory_row])
              transition_row = np.concatenate([transition_row, stream_row])

              row_tup = (s_e, s_h, s_m, s_c_active)
              row_index = self.state_tuple_to_index[row_tup]

              # catch OOB probabilities due to input data and renormalize
              transition_row[transition_row < 0] = 0
              transition_row = transition_row/transition_row.sum()

              transition_probs_one_arm[row_index, a] = transition_row

      self.transition_probs[arm_i] = transition_probs_one_arm

  def reset_states(self) -> np.ndarray:
    """Set all states to engaged and A1c > 8, or proper streaming position."""

    # first find all the streaming positions
    unique_streaming_starts = sorted(set(self.stream_map))

    for streaming_start_position in unique_streaming_starts:

      # if starting at beginning, give an active start state
      if streaming_start_position == 0:
        s_e = self.ENGAGED
        s_h = self.A1CGT8
        s_m = (self.MAINTENANCE, self.MAINTENANCE)
        s_c = self.ARM_ACTIVE
        state_tup = (s_e, s_h, s_m, s_c)
      else:
        s_e = self.engagement_states[0]
        s_h = self.health_states[0]
        s_m = self.memory_states[0]

        state_tup = (s_e, s_h, s_m, streaming_start_position)

      state_i = self.state_tuple_to_index[state_tup]
      self.current_states[self.stream_map == streaming_start_position] = state_i

    return self.current_states

  def set_reward_definition(self) -> None:
    """Set the reward definition."""
    streaming_inactive_reward = 0

    for n in range(self.n_arms):
      group = self.group_map[n]
      engagement_rewards = {
          self.DROPOUT: self.group_rewards[group]['r_dropout'],
          self.MAINTENANCE: self.group_rewards[group]['r_maintenance'],
          self.ENGAGED: self.group_rewards[group]['r_eng'],
      }

      health_rewards = {
          self.A1CGT8: self.group_rewards[group]['r_a1cgt8'],
          self.A1CLT8: self.group_rewards[group]['r_a1clt8']
      }

      # first add streaming reward definitions
      # loop backwards from latest possible start
      for s_c in range(self.stream_horizon-1, 0, -1):
        # setting up streaming counter states, define some default values
        s_e = self.engagement_states[0]
        s_h = self.health_states[0]
        s_m = self.memory_states[0]

        state_tup = (s_e, s_h, s_m, s_c)
        state_ind = self.state_tuple_to_index[state_tup]
        self.rewards[n, state_ind] = streaming_inactive_reward
        reward_tuple = (streaming_inactive_reward, streaming_inactive_reward)
        self.rewards_2d[n, state_ind] = reward_tuple

      s_c_active = self.ARM_ACTIVE

      for s_e in self.engagement_states:
        for s_h in self.health_states:
          for s_m in self.memory_states:
            state_tup = (s_e, s_h, s_m, s_c_active)
            state_ind = self.state_tuple_to_index[state_tup]
            state_reward = self.alpha * engagement_rewards[s_e] + (
                1 - self.alpha) * health_rewards[s_h]
            self.rewards[n, state_ind] = state_reward
            reward_tuple = (engagement_rewards[s_e], health_rewards[s_h])
            self.rewards_2d[n, state_ind] = reward_tuple


class DiabetesAppPartialObsEnvironment(RMABEnvironment):
  """Class for the Diabetes App Environment with partial observability."""

  env_key_name = 'Diabetes App Partially Observable Environment'

  MAINTENANCE = 0
  ENGAGED = 1

  A1CGT8 = 0
  A1CLT8 = 1

  NODROPOUT = 0
  DROPOUT = 1

  # Value of s_t when arm gives an observation
  OBSERVE_RESET = 0

  # Value of s_counter when arm becomes active (streaming)
  ARM_ACTIVE = 0

  # index position of each state within the state tuple
  HEALTH_INDEX = 4
  ACTIVE_INDEX = 5

  def __init__(self,
               n_arms: int,
               start_seed: int,
               horizon: int,
               model_data_file: str,
               stream_map: np.ndarray,
               alpha: float = 0.5,
               ) -> None:
    """Initialize the DiabetesAppPartialObsEnvironment class.

    Args:
      n_arms: number of arms in the RMAB
      start_seed: starting random seed for experiments
      horizon: length of simulation
      model_data_file: file with data to populate group_probs, group_map,
        and group_rewards
      stream_map: defines when each arm enters the simulation
      alpha: weight to place on engagement rewards (remaining weight goes on
        health rewards)
    """
    super().__init__(n_arms, start_seed, horizon, model_data_file, stream_map)

    data = DiabetesAppPartialObsEnvironment.parse_model_data(
        model_data_file,
        n_arms
    )
    self.group_probs = data[0]
    self.group_map = data[1]
    self.group_definitions = data[2]
    self.group_rewards = data[3]

    self.n_actions = 2
    self.horizon = horizon
    self.alpha = alpha
    self.stream_map = stream_map

    # if no streaming is defined, all arms start on round 0
    if self.stream_map is None:
      self.stream_map = np.zeros(self.n_arms)

    # if no group rewards defined, use these defaults
    if self.group_rewards is None:
      self.group_rewards = []
      for group in set(self.group_map):
        self.group_rewards[group] = {
            'r_dropout': 0,
            'r_maintenance': 1,
            'r_eng': 1,
            'r_a1cgt8': 0,
            'r_a1clt8': 1,
        }

    # slightly changed state definition to handle partial observability
    self.state_dimension_counts = [
        2,  # engagement states (Maintenance, Engaged)
        2,  # Health states (A1c >8, A1c < 8)
        4,  # Memory states ([m, m], [m, e], ...)
        horizon,  # time dimension
        2,  # dropout states (not dropout, dropout)
    ]

    self.engagement_states = np.arange(self.state_dimension_counts[0])
    self.health_states = np.arange(self.state_dimension_counts[1])
    self.memory_states = list(
        itertools.product(self.engagement_states, repeat=2))
    self.time_states = np.arange(self.state_dimension_counts[3])
    self.dropout_states = np.arange(self.state_dimension_counts[4])

    # streaming states will be a linear addition to the state space count
    self.streaming_states = np.arange(horizon)

    self.n_states = math.prod(self.state_dimension_counts)
    self.n_states += len(self.streaming_states)-1

    # create the combined state reference dictionaries
    ret_data = self.get_state_dicts_combined()
    self.state_tuple_to_index = ret_data[0]
    self.state_index_to_tuple = ret_data[1]

    # create the state reference dictionaries for the memory dimension
    ret_data = self.get_state_dicts_memory()
    self.state_tuple_to_index_mem = ret_data[0]
    self.state_index_to_tuple_mem = ret_data[1]

    # initialize transition probabilities
    self.transition_probs = np.zeros(
        (self.n_arms, self.n_states, self.n_actions, self.n_states))
    self.get_transition_probabilities()

    # initialize state
    # design choice: states will be kept and returned as indexes
    self.current_states = np.zeros(self.n_arms, dtype=int)
    self.reset_states()

    # define rewards
    self.rewards = np.zeros((self.n_arms, self.n_states), dtype=np.float)
    self.rewards_2d = np.zeros((self.n_arms, self.n_states, 2), dtype=np.float)
    self.set_reward_definition()

  @staticmethod
  def env_key() -> str:
    """Return the name of the environment."""
    return DiabetesAppPartialObsEnvironment.env_key_name

  @staticmethod
  def environment_params() -> collections.OrderedDict[str, str]:
    """Return an OrderedDict of environment hyperparameters and their types."""
    params = collections.OrderedDict()
    params['alpha'] = 'float'
    return params

  @staticmethod
  def parse_model_data(filename, n_arms):
    return DiabetesAppEnvironment.parse_model_data(filename, n_arms)

  def create_active_arms_helper(self):
    self.active_arms_helper = util.GetActiveArmsDiabetesApp(
        self.state_index_to_tuple,
        self.ARM_ACTIVE,
        self.ACTIVE_INDEX,
    )
    return self.active_arms_helper

  # def get_active_arms(self, states):
  #   """Method to get active arms of this environment."""
  #   return self.active_arms_helper.get_active_arms(states)
  #   # active_arms = np.zeros(states.shape[0], dtype=int)
  #   # for i, s in enumerate(states):
  #   #   state_tup = self.state_index_to_tuple[s]
  #   #   s_c = state_tup[5]
  #   #   if s_c == 0:
  #   #     active_arms[i] = 1
  #   # return active_arms

  def get_info(self) -> dict[str, object]:
    """Return the state index-to-tuple translators."""
    return {
        'state_index_to_tuple': self.state_index_to_tuple,
        'state_index_to_tuple_mem': self.state_index_to_tuple_mem,
        'state_tuple_to_index': self.state_tuple_to_index,
        'state_tuple_to_index_mem': self.state_tuple_to_index_mem,
        'rewards_2d': self.rewards_2d,
        'alpha': self.alpha,
        'high_a1c_code': self.A1CGT8,
        'state_tuple_health_index': self.HEALTH_INDEX,
        'group_map': self.group_map,
        'active_arms_helper': self.create_active_arms_helper()
    }

  def get_inactive_state(self, s_c):
    s_e = self.engagement_states[0]
    s_m = self.memory_states[0]
    s_t = self.time_states[0]
    s_d = self.dropout_states[0]
    s_h = self.health_states[0]
    return (s_e, s_m, s_t, s_d, s_h, s_c)

  def get_state_dicts_memory(
      self) -> (dict[tuple[int, int], int], dict[int, tuple[int, int]]):
    """Create the state reference dictionaries for the memory dimension."""

    index_to_tuple = dict(list(enumerate(self.memory_states)))
    tuple_to_index = dict([
        (value, key) for key, value in index_to_tuple.items()
    ])
    return tuple_to_index, index_to_tuple

  def get_state_dicts_combined(
      self) -> (dict[tuple[object], int], dict[int, tuple[object]]):
    """Create the state reference dictionaries over all states."""
    all_active_tuples = list(
        itertools.product(
            self.engagement_states,
            self.memory_states,
            self.time_states,
            self.dropout_states,
            self.health_states,
            [self.ARM_ACTIVE],  # stream counter is 0 when arm is active
            repeat=1))

    # setting up streaming counter states, define some default values
    s_e = self.engagement_states[0]
    s_m = self.memory_states[0]
    s_t = self.time_states[0]
    s_d = self.dropout_states[0]
    s_h = self.health_states[0]
    stream_tuples = [
        (s_e, s_m, s_t, s_d, s_h, s_c) for s_c in range(1, self.horizon)
    ]

    # add stream counter states to the end
    all_tuples = all_active_tuples + stream_tuples

    index_to_tuple = dict(list(enumerate(all_tuples)))
    tuple_to_index = dict([
        (value, key) for key, value in index_to_tuple.items()
    ])
    return tuple_to_index, index_to_tuple

  def sample_and_store(self, prob_name, arm):
    """Sample transition probabilities with some noise."""
    tiny_eps = 1e-3
    std = 0.05
    group = self.group_map[arm]
    prob = self.group_probs[group][prob_name]
    sampled_prob = min(1-tiny_eps, self.random_stream.normal(prob, std))
    sampled_prob = max(tiny_eps, sampled_prob)
    self.arm_probs[arm][prob_name] = sampled_prob
    if prob < 0.025:
      sampled_prob = prob  # BUG: this creates a small error in the way
      # the bayesian rewards are computed... but max error of the bug is
      # small TODO: Fix and test.
      # Then replace with the sample_and_store implementation of the
      # fully observable diabetes app env
    return sampled_prob

  # def sample_and_store(self, prob_name, arm):
  #   """Sample transition probabilities with some noise."""
  #   tiny_eps = 1e-3
  #   std = 0.1

  #   group = self.group_map[arm]
  #   prob = self.group_probs[group][prob_name]

  #   # want to scale deviations based on their distance from 0 or 1
  #   scale = min(abs(prob), abs(1-prob))

  #   # introduce small variation between arms in each group
  #   sigma = std*scale
  #   sampled_prob = min(1-tiny_eps, self.random_stream.normal(prob, sigma))
  #   sampled_prob = max(tiny_eps, sampled_prob)
  #   self.arm_probs[arm][prob_name] = sampled_prob

  #   return sampled_prob

  def get_transition_probabilities(self) -> None:
    """Create the app-engagement/diabetes progression RMAB."""

    self.arm_probs = [{} for arm in range(self.n_arms)]

    for arm_i in range(self.n_arms):

      p_i_mtoe = self.sample_and_store('p_i_mtoe', arm_i)
      p_i_mtod = self.sample_and_store('p_i_mtod', arm_i)
      p_u_mtod = self.sample_and_store('p_u_mtod', arm_i)
      p_i_etoe = self.sample_and_store('p_i_etoe', arm_i)

      p_eng_gl = self.sample_and_store('p_eng_gl', arm_i)
      p_eng_ll = self.sample_and_store('p_eng_ll', arm_i)
      p_noeng_gl = self.sample_and_store('p_noeng_gl', arm_i)
      p_noeng_ll = self.sample_and_store('p_noeng_ll', arm_i)

      self.p_eng_observe = 1.0  # assumes full observability on eng state
      self.p_mnt_observe = 0.3
      self.p_do_observe = 0.0

      transition_probs_one_arm = np.zeros(
          (self.n_states, self.n_actions, self.n_states))

      # The workflow for defining transition probabilities is as follows:
      # 1. First, streaming states are "tacked on", since they are
      #    disjoint from the rest of the state space evolution.
      # 2. To implement the partially observable environment as an MDP
      #    we carry out a transformation to a belief-state MDP. Moreover, we
      #    have special observability structure, in which state is either fully
      #    observed or not with some known probability. In other words, there is
      #    some state-dependent probability of receiving an observation from the
      #    model, and if an observation is received,  we know its state with
      #    certainty. This combined with a finite horizon means the belief-stat
      #    MDP is finite, and can be defined by finite "chains", where the head
      #    of a chain is a state that can be fully observed, and each subsequent
      #    node in the chain corresponds to the number of time periods that have
      #    passed since the state was observed. See the forthcoming publication
      #    for complete mathematical details of how the conversion is carried
      #    out.
      #    With respect to the implementation here, we define probabilities
      #    using this chain logic, i.e., checking (1) checking the states of the
      #    head of the chain and (2) the position of the current state within
      #    the chain, to determine the probability of transition to the next
      #    state.
      # Note: This class is a specialized implementation for a specific use-case
      #    that trades off interpretability for computational scalability.
      #    If you are just getting started with this repository and want to
      #    better understand this environment at a high level (e.g., for its
      #    joint engagement/health state dynamics) please see
      #    DiabetesAppEnvironment, which is the much simpler and fully
      #    observable equivalent of this class. Alternatively, to better
      #    understand the basics of a working RMABEnvironment, please see
      #    RandomEnvironment.

      def engaged_to_engaged(prob_row, health_p=1, s_m_next=()):
        # reset to the head of appropriate chains

        # Next is (engaged, s_m, observe, no dropout, A1CLT8)
        next_tup = (self.ENGAGED, s_m_next, self.OBSERVE_RESET, self.NODROPOUT,
                    self.A1CLT8, self.ARM_ACTIVE)
        ind = self.state_tuple_to_index[next_tup]
        # pylint: disable=cell-var-from-loop
        prob_row[ind] = p_i_etoe * health_p

        # Next is (engaged, s_m, observe, no dropout, A1CGT8)
        next_tup = (self.ENGAGED, s_m_next, self.OBSERVE_RESET, self.NODROPOUT,
                    self.A1CGT8, self.ARM_ACTIVE)
        ind = self.state_tuple_to_index[next_tup]
        prob_row[ind] = p_i_etoe * (1 - health_p)

      def engaged_to_maintenance(prob_row,
                                 eng_p=1,
                                 health_p=1,
                                 s_m_next=(),
                                 state_tup=()):
        # Next is (maint, observe, A1CLT8)
        # reset to the head of appropriate chain
        next_tup = (self.MAINTENANCE, s_m_next, self.OBSERVE_RESET,
                    self.NODROPOUT, self.A1CLT8, self.ARM_ACTIVE)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = (eng_p) * (self.p_mnt_observe) * (health_p)

        # Next is (maint, observe, A1CGT8)
        # reset to the head of appropriate chain
        next_tup = (self.MAINTENANCE, s_m_next, self.OBSERVE_RESET,
                    self.NODROPOUT, self.A1CGT8, self.ARM_ACTIVE)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = (eng_p) * (self.p_mnt_observe) * (1 - health_p)

        # Next is (maint, no observe)
        # just step down the chain
        s_e = state_tup[0]
        s_m = state_tup[1]
        s_t = min(state_tup[2] + 1, self.horizon - 1)
        s_d = state_tup[3]
        s_h = state_tup[4]
        s_c = self.ARM_ACTIVE
        next_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = (eng_p) * (1 - self.p_mnt_observe)

      def from_maintenance_to_engaged(prob_row, eng_p=1,
                                      health_p=1, s_m_next=()):
        # Next is (engaged, s_m, observe, no dropout, A1CLT)
        # reset to the head of appropriate chain
        next_tup = (self.ENGAGED, s_m_next, self.OBSERVE_RESET, self.NODROPOUT,
                    self.A1CLT8, self.ARM_ACTIVE)
        ind = self.state_tuple_to_index[next_tup]
        prob_row[ind] = eng_p * health_p

        # Next is (engaged, s_m, observe, no dropout, A1CGT)
        # reset to the head of appropriate chain
        next_tup = (self.ENGAGED, s_m_next, self.OBSERVE_RESET, self.NODROPOUT,
                    self.A1CGT8, self.ARM_ACTIVE)
        ind = self.state_tuple_to_index[next_tup]
        prob_row[ind] = eng_p * (1 - health_p)

      def from_maintenance_to_maintenance(prob_row,
                                          eng_p=1,
                                          health_p=1,
                                          s_m_next=(),
                                          state_tup=()):
        # Next is (maint, s_m, observe, no dropout, A1CLT)
        # reset to the head of appropriate chain
        next_tup = (self.MAINTENANCE, s_m_next, self.OBSERVE_RESET,
                    self.NODROPOUT, self.A1CLT8, self.ARM_ACTIVE)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = eng_p * (self.p_mnt_observe) * health_p

        # Next is (maint, s_m, observe, no dropout, A1CGT)
        # reset to the head of appropriate chain
        next_tup = (self.MAINTENANCE, s_m_next, self.OBSERVE_RESET,
                    self.NODROPOUT, self.A1CGT8, self.ARM_ACTIVE)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = eng_p * (self.p_mnt_observe) * (1 - health_p)

        # Next is (maint, no observe)
        # just step down the chain
        s_e = state_tup[0]
        s_m = state_tup[1]
        s_t = min(state_tup[2] + 1, self.horizon - 1)
        s_d = state_tup[3]
        s_h = state_tup[4]
        s_c = self.ARM_ACTIVE
        next_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = (eng_p) * (1 - self.p_mnt_observe)

      def from_maintenance_to_dropout(prob_row, eng_p=1, state_tup=()):

        # Next is (maint, no observe, dropout)
        # just step down the chain add dropout
        s_e = state_tup[0]
        s_m = state_tup[1]
        s_t = min(state_tup[2] + 1, self.horizon - 1)
        s_d = self.DROPOUT
        s_h = state_tup[4]
        s_c = self.ARM_ACTIVE
        next_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = eng_p

      def step_down_chain(prob_row, state_tup=()):
        # Next is (dropout, no observe)
        # just step down the chain
        s_e = state_tup[0]
        s_m = state_tup[1]
        s_t = min(state_tup[2] + 1, self.horizon - 1)
        s_d = state_tup[3]
        s_h = state_tup[4]
        s_c = self.ARM_ACTIVE
        next_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        index = self.state_tuple_to_index[next_tup]
        prob_row[index] = 1

      def exit_streaming_start_distribution(prob_row):
        # for now, have all arms start in engaged, mem=[M, M] high A1c
        s_e = self.ENGAGED
        s_m = (self.MAINTENANCE, self.MAINTENANCE)
        s_t = self.OBSERVE_RESET
        s_d = self.NODROPOUT
        s_h = self.A1CGT8
        s_c = self.ARM_ACTIVE
        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        index = self.state_tuple_to_index[state_tup]
        prob_row[index] = 1

      # first add streaming transition definitions
      # loop backwards from latest possible start
      for s_c in range(self.horizon-1, 0, -1):
        # setting up streaming counter states, define some default values
        s_e = self.engagement_states[0]
        s_m = self.memory_states[0]
        s_t = self.time_states[0]
        s_d = self.dropout_states[0]
        s_h = self.health_states[0]
        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)

        transition_row = np.zeros(self.n_states)

        # count down
        if s_c > 1:
          next_state_tup = (s_e, s_m, s_t, s_d, s_h, s_c-1)
          index = self.state_tuple_to_index[next_state_tup]
          transition_row[index] = 1
        elif s_c == 1:
          exit_streaming_start_distribution(transition_row)

        # done, now set the transition probability distro for this state
        state_index = self.state_tuple_to_index[state_tup]

        # streaming transitions are same for all actions
        transition_probs_one_arm[state_index, :] = transition_row

      s_c_active = self.ARM_ACTIVE

      # Then, define the environment itself for active arms
      for s_e in self.engagement_states:
        for s_m in self.memory_states:
          for s_h in self.health_states:
            for s_d in self.dropout_states:
              belief_a1c_lt8 = s_h
              for s_t in self.time_states:
                for a in range(self.n_actions):

                  engagement_row = np.zeros(self.state_dimension_counts[0])
                  health_row = np.zeros(self.state_dimension_counts[1])
                  memory_row = np.zeros(self.state_dimension_counts[2])
                  time_row = np.zeros(self.state_dimension_counts[3])
                  dropout_row = np.zeros(self.state_dimension_counts[4])
                  stream_row = np.zeros(self.horizon - 1)

                  state_tup = (s_e, s_m, s_t, s_d, s_h, s_c_active)

                  transition_row = functools.reduce(np.kron, [
                      engagement_row, memory_row, time_row, dropout_row,
                      health_row
                  ])
                  transition_row = np.concatenate([transition_row, stream_row])

                  # In this partially observable world, dynamics driven by
                  # chains and counters.
                  # The head of the chain is all possible combinations of
                  # current states and memory states,
                  # e.g., [M, M, M], [M, M, E], ...,
                  # so there are 8 possible chain heads.
                  # if we are in s_t={0,1,2}, we check these states
                  # according to how much time has passed...
                  # e.g., if s_t==2, then three time periods haved passed
                  # since the last observation... so we check s_e to see how
                  # health should evolve according to behavior 3 months ago.
                  if s_t == 0:
                    next_mem_tup = (s_e, s_m[0])  # advance memory by 1
                    eng_state_to_check = s_e  # check current engagement
                    health_state_to_check = s_m[
                        1]  # check 3-months ago engagement

                  # Note: if s_t>=1, arm was not ENGAGED
                  elif s_t == 1:
                    # advance memory by 2
                    next_mem_tup = (self.MAINTENANCE, s_e)
                    # if s_t>=1, was not eng
                    eng_state_to_check = self.MAINTENANCE
                    health_state_to_check = s_m[
                        0]  # check 3-months ago engagement
                  elif s_t == 2:
                    next_mem_tup = (self.MAINTENANCE, self.MAINTENANCE)
                    # if s_t>=1, was not eng
                    eng_state_to_check = self.MAINTENANCE
                    # check 3-months ago engagement
                    health_state_to_check = s_e
                  elif s_t >= 3:
                    next_mem_tup = (self.MAINTENANCE, self.MAINTENANCE)
                    eng_state_to_check = self.MAINTENANCE
                    health_state_to_check = self.MAINTENANCE

                  # A1c rules.
                  # probability that we go to A1CLT8 next round is same as a
                  # single belief state update
                  # according to this round's health_state_to_check... so:
                  if health_state_to_check == self.ENGAGED:
                    p_gl = p_eng_gl
                    p_ll = p_eng_ll
                  elif health_state_to_check != self.ENGAGED:
                    p_gl = p_noeng_gl
                    p_ll = p_noeng_ll
                  belief_a1c_lt8 = self.evolve_belief(belief_a1c_lt8,
                                                      p_gl, p_ll)

                  # Engagement rules: dropout rules handled within the functions
                  if eng_state_to_check == self.ENGAGED:
                    if a == 1:
                      engaged_to_engaged(
                          transition_row,
                          health_p=belief_a1c_lt8,
                          s_m_next=next_mem_tup)
                      engaged_to_maintenance(
                          transition_row,
                          eng_p=(1 - p_i_etoe),
                          health_p=belief_a1c_lt8,
                          s_m_next=next_mem_tup,
                          state_tup=state_tup)

                    elif a == 0:
                      # always go to maintenance
                      engaged_to_maintenance(
                          transition_row,
                          eng_p=1,
                          health_p=belief_a1c_lt8,
                          s_m_next=next_mem_tup,
                          state_tup=state_tup)

                  # then define rewards and state resets...
                  elif eng_state_to_check == self.MAINTENANCE:
                    if s_d == self.DROPOUT:
                      # if in dropout, can only advance down the current chain
                      # and cannot leave dropout
                      step_down_chain(transition_row, state_tup=state_tup)

                    elif s_d == self.NODROPOUT:

                      if a == 1:
                        from_maintenance_to_engaged(
                            transition_row,
                            eng_p=p_i_mtoe,
                            health_p=belief_a1c_lt8,
                            s_m_next=next_mem_tup)
                        from_maintenance_to_maintenance(
                            transition_row,
                            eng_p=(1 - p_i_mtod - p_i_mtoe),
                            health_p=belief_a1c_lt8,
                            s_m_next=next_mem_tup,
                            state_tup=state_tup)
                        from_maintenance_to_dropout(
                            transition_row,
                            eng_p=p_i_mtod,
                            state_tup=state_tup)

                      elif a == 0:
                        from_maintenance_to_maintenance(
                            transition_row,
                            eng_p=(1 - p_u_mtod),
                            health_p=belief_a1c_lt8,
                            s_m_next=next_mem_tup,
                            state_tup=state_tup)
                        from_maintenance_to_dropout(
                            transition_row,
                            eng_p=p_u_mtod,
                            state_tup=state_tup)

                  # Now set the transition probability distro for this state
                  state_index = self.state_tuple_to_index[state_tup]

                  # Catch OOB probabilities due to input data and renormalize
                  transition_row[transition_row < 0] = 0
                  transition_row = transition_row/transition_row.sum()

                  transition_probs_one_arm[state_index, a] = transition_row

      # Done creating single transition matrix.
      self.transition_probs[arm_i] = transition_probs_one_arm

  def reset_states(self) -> np.ndarray:
    """Set all states to engaged and A1c > 8, or proper streaming position."""

    # first find all the streaming positions
    unique_streaming_starts = sorted(set(self.stream_map))

    for streaming_start_position in unique_streaming_starts:

      # if starting at beginning, give an active start state
      if streaming_start_position == 0:
        s_e = self.ENGAGED
        s_m = (self.MAINTENANCE, self.MAINTENANCE)
        s_t = self.OBSERVE_RESET
        s_d = self.NODROPOUT
        s_h = self.A1CGT8
        s_c = self.ARM_ACTIVE
        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
      else:
        s_e = self.engagement_states[0]
        s_m = self.memory_states[0]
        s_t = self.time_states[0]
        s_d = self.dropout_states[0]
        s_h = self.health_states[0]
        state_tup = (s_e, s_m, s_t, s_d, s_h, streaming_start_position)

      state_i = self.state_tuple_to_index[state_tup]
      self.current_states[self.stream_map == streaming_start_position] = state_i

    return self.current_states

  def evolve_belief(self, belief, p_gl, p_ll):
    # belief update is:
    # prob(greater than 8)*prob(going to less than 8) +
    #    prob(less than 8)*prob(stay less than 8)
    new_belief = (1 - belief) * p_gl + belief * (p_ll)

    return new_belief

  def get_reward_definition_2d(self):
    return np.copy(self.rewards_2d)

  def set_reward_definition(self) -> None:
    """Set the reward definition."""

    streaming_inactive_reward = 0

    for n in range(self.n_arms):
      group = self.group_map[n]
      engagement_rewards = {
          # self.MAINTENANCE: 0.5,
          self.MAINTENANCE: self.group_rewards[group]['r_maintenance'],
          self.ENGAGED: self.group_rewards[group]['r_eng'],
      }

      # rules for dropout reward will happen in the if-ladder
      dropout_reward = self.group_rewards[group]['r_dropout']

      health_rewards = {
          self.A1CGT8: self.group_rewards[group]['r_a1cgt8'],
          self.A1CLT8: self.group_rewards[group]['r_a1clt8'],
      }

      # first add streaming reward definitions
      # loop backwards from latest possible start
      for s_c in range(self.horizon-1, 0, -1):
        # setting up streaming counter states, define some default values
        s_e = self.engagement_states[0]
        s_m = self.memory_states[0]
        s_t = self.time_states[0]
        s_d = self.dropout_states[0]
        s_h = self.health_states[0]

        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        state_ind = self.state_tuple_to_index[state_tup]
        self.rewards[n, state_ind] = streaming_inactive_reward
        reward_tuple = (streaming_inactive_reward, streaming_inactive_reward)
        self.rewards_2d[n, state_ind] = reward_tuple

      s_c_active = self.ARM_ACTIVE
      # Then define environment rewards
      for s_e in self.engagement_states:
        for s_m in self.memory_states:
          for s_h in self.health_states:
            for s_d in self.dropout_states:
              health_belief = s_h
              for s_t in self.time_states:

                # Need to do belief state evolution for the health rewards
                # this is as simple as checking s_h, the initial three
                # engagement states, and current time on chain, then evolving
                # the belief state to get appropriate reward.

                # Engagement rewards
                engagement_reward = 0
                if s_t == 0:
                  eng_state_to_check = s_e
                  engagement_reward = engagement_rewards[eng_state_to_check]
                  # can't be in dropout at head of chain
                elif s_t >= 1:
                  if s_d == self.DROPOUT:
                    engagement_reward = dropout_reward
                  elif s_d == self.NODROPOUT:
                    eng_state_to_check = self.MAINTENANCE  # if s_t>1, not eng
                    engagement_reward = engagement_rewards[eng_state_to_check]

                # Health rewards equal belief arm is in A1c < 8 health state
                if s_t == 0:
                  health_belief = s_h
                elif s_t >= 1:
                  # evolve state according to the 3-months ago engagement
                  if s_t == 1:
                    health_state_to_check = s_m[1]
                  elif s_t == 2:
                    health_state_to_check = s_m[0]
                  elif s_t == 3:
                    health_state_to_check = s_e
                  elif s_t > 3:
                    health_state_to_check = self.MAINTENANCE

                  if health_state_to_check == self.ENGAGED:
                    p_gl = self.arm_probs[n]['p_eng_gl']
                    p_ll = self.arm_probs[n]['p_eng_ll']
                  elif health_state_to_check != self.ENGAGED:
                    p_gl = self.arm_probs[n]['p_noeng_gl']
                    p_ll = self.arm_probs[n]['p_noeng_ll']

                  health_belief = self.evolve_belief(health_belief, p_gl, p_ll)

                health_reward = health_rewards[self.A1CGT8] * (
                    1 -
                    health_belief) + health_rewards[self.A1CLT8] * health_belief

                state_tup = (s_e, s_m, s_t, s_d, s_h, s_c_active)
                state_ind = self.state_tuple_to_index[state_tup]
                self.rewards[n, state_ind] = self.alpha * engagement_reward + (
                    1 - self.alpha) * health_reward
                reward_tuple = (engagement_reward, health_reward)
                self.rewards_2d[n, state_ind] = reward_tuple


class Config():
  """Class for getting references to and instantiating RMABEnvironments."""

  # Define policy class mapping
  # If defining a new class, add its name to class_list
  class_list = [
      RandomEnvironment,
      TwoStateEnvironment,
      MaternalHealthEnvironment,
      DiabetesAppEnvironment,
      DiabetesAppPartialObsEnvironment
  ]
  env_class_map = {cl.env_key(): cl for cl in class_list}

  @staticmethod
  def get_env_class(env_name):
    """Return a class reference corresponding the input environment string."""
    try:
      env_class = Config.env_class_map[env_name]
    except KeyError:
      err_string = 'Did not recognize environment "%s". Please add it to '
      err_string += 'environments.Config.class_list.'
      print(err_string % env_name)
      exit()
    return env_class


