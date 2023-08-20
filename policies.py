"""Defines policies for acting in RMAB disease environments.

Contains classes that define policies for acting in RMAB
environments defined in environments.py.

Author: killian-34/jakillian
"""

import abc  # create abstract base classes
import os
import tempfile
import time
from typing import Any

import mathprog_methods
import mdptoolbox
import numpy as np
import pulp
import ujson
import util


class RMABPolicy(abc.ABC):
  """Abstract class for RMAB policies."""

  policy_key_str = 'RMAB Policy'

  def __init__(self,
               n_arms: int,
               budget: int = None,
               transition_probs: np.ndarray = None,
               rewards: np.ndarray = None,
               **kwargs,
               ) -> None:
    """Initialize the abstract RMABpolicy class.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      **kwargs: additional keyword arguments
    """
    self.n_arms = n_arms
    self.budget = budget
    self.transition_probs = transition_probs
    self.rewards = rewards
    self.kwargs = kwargs

  def __repr__(self) -> str:
    """Return the string representation of the class."""
    return self.policy_key()

  @abc.abstractmethod
  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Return actions to take in the RMAB.

    Args:
      current_states: current state of the arms in the RMAB
      remaining_horizon: time periods remaining in the simulation

    Returns:
      actions: n_arms-length integer array of actions to take in the RMAB
    """
    pass

  @staticmethod
  @abc.abstractmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    raise NotImplementedError

  def signal_reset(self):
    """If a learning policy, inform policy when environment resets."""
    pass


class RandomPolicy(RMABPolicy):
  """Class for implementing the Random RMAB policy (binary-action)."""

  policy_key_str = 'Random Policy'

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return RandomPolicy.policy_key_str

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Get random actions the satisfy the budget.

    Args:
      current_states: unused, but required in the abstract class, since
        intelligent policies will use it.
      remaining_horizon: unused, but required in the abstract class, since
        intelligent policies will use it.

    Returns:
      random_actions: n_arms-length array of random actions

    """
    random_actions = np.zeros(self.n_arms, dtype=np.int32)
    choices = np.random.choice(
        a=np.arange(self.n_arms), size=self.budget, replace=False)
    random_actions[choices] = 1
    return random_actions


class WhittleIndexPolicy(RMABPolicy):
  """Class for implementing the Whittle Index Policy."""

  policy_key_str = 'Whittle Index Policy'

  def __init__(self,
               n_arms: int,
               budget: int,
               transition_probs: np.ndarray,
               rewards: np.ndarray,
               index_lb: float = -1.0,
               index_ub: float = 1.0,
               gamma: float = 1.0,
               binary_search_tolerance: float = 1e-1,
               max_horizon: int = 20,
               **kwargs,
               ) -> None:
    """Initialize the WhittleIndexPolicy class.

    Contains code for computing the
    Whittle index policy for a given RMAB.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      index_lb: lower bound for initial Whittle index estimate (binary search)
      index_ub: upper bound for initial Whittle index estimate (binary search)
      gamma: discount factor for computing the Whittle index (binary search)
      binary_search_tolerance: maximum allowable error when computing the
        Whittle index (binary search)
      **kwargs: additional keyword arguments
    """
    super().__init__(
        n_arms,
        budget=budget,
        transition_probs=transition_probs,
        rewards=rewards,
        **kwargs,
        )
    self.index_lb = index_lb
    self.index_ub = index_ub
    self.gamma = gamma
    self.binary_search_tolerance = binary_search_tolerance
    self.tie_break_epsilon = 1e-4
    self.max_horizon = max_horizon

    # Set self.do_initial_search to true if you do not know the bounds
    # on your Whittle indexes -- it may slightly increase runtime, but
    # it guarantees that you will find the correct Whittle index value
    self.do_initial_search = False

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return WhittleIndexPolicy.policy_key_str

  def binary_search_for_whittle_index(self, t_probs, r, state: int,
                                      remaining_horizon: int) -> float:
    """Perform binary search to find the Whittle index.

    This version
    will first search for confirmed upper and lower bounds before
    carrying out the binary search to ensure validity of the
    Whittle index estimate.

    Args:
      t_probs: transition probabilities for one arm (SxAxS)
      r: reward function for one arm (S)
      state: state of the arm for which to search for the index
      remaining_horizon: number of time periods remaining in the simulation

    Returns:
      wi_estimate: approximation of the Whittle index, to
      self.binary_search_tolerance level
        of precision

    """

    num_actions = 2  # whittle index only defined for two-action settings
    num_states = t_probs.shape[0]

    # Change from S,A,S format to A,S,S format (required by pymdptoolbox)
    t_array = np.swapaxes(t_probs, 0, 1)

    # Rewards also need to be in A,S,S format too
    # (the rest of the repo assumes rewards are only function of current state)
    r_array_base = np.zeros(t_array.shape)
    for a in range(num_actions):
      for s in range(num_states):
        r_array_base[a, :, s] += r

    # First, check to see if we should go up or down from the
    # current whittle index estimate.

    upper = self.index_ub
    lower = self.index_lb

    wi_estimate = (upper + lower) / 2

    if self.do_initial_search:
      _, action = self.solve_adjusted_mdp(
          state, r_array_base, t_array, wi_estimate, remaining_horizon
      )

      # If optimal to be passive, index (penalty) is too high, so search lower
      if action == 0:
        upper = wi_estimate

      # If optimal to act, index (penalty) is too low, so search higher
      elif action == 1:
        lower = wi_estimate

      # Now search that direction until optimal action flips,
      # establishing our bounds
      previous_action = action
      while action == previous_action:

        # needs to be here for loop logic
        previous_action = action

        # if not acting, reduce index estimate
        if previous_action == 0:
          wi_estimate = lower
        elif previous_action == 1:
          wi_estimate = upper

        _, action = self.solve_adjusted_mdp(
            state, r_array_base, t_array, wi_estimate, remaining_horizon
        )

        # if optimal action is still the same, keep searching this direction
        if action == previous_action:
          # if passive, reduce index (action penalty), reduce the lower bound
          if action == 0:
            upper = lower
            lower = self.reduce_lb(lower)
          # if acting, do the converse
          elif action == 1:
            lower = upper
            upper = self.increase_ub(upper)

        # If optimal action has changed, we have found the bounds
        # for our binary search.
        else:
          if action == 0:
            upper = wi_estimate
          elif action == 1:
            lower = wi_estimate

          # the while loop condition should then break appropriately
          # if action has changed

      # s2 = time.time()
      # print('BS search time',s2-s1)
      # print(upper, lower)
    # s2 = time.time()

    # Now carry out the binary search with the confirmed bounds
    while (upper - lower) > self.binary_search_tolerance:
      wi_estimate = (upper + lower) / 2
      # sa = time.time()
      _, action = self.solve_adjusted_mdp(
          state, r_array_base, t_array, wi_estimate, remaining_horizon
      )
      # sb = time.time()
      # print('solvetime',sb-sa)
      if action == 0:
        upper = wi_estimate
      elif action == 1:
        lower = wi_estimate

    wi_estimate = (upper + lower) / 2
    # s3 =time.time()

    # print('BS actual time',s3-s2)
    return wi_estimate

  def reduce_lb(self, lb: int) -> int:
    """Easy rules for decreasing lower bound near 0, and for each sign.

    Args:
      lb: the bound to lower

    Returns:
      lb_lower: the lowered bound
    """
    if lb <= -1:
      return lb * 2
    elif lb > -1 and lb < 1:
      return lb - 1
    elif lb >= 1:
      return lb / 2

  def increase_ub(self, ub: int) -> int:
    """Easy rules for increasing upper bound near 0, and for each sign.

    Args:
      ub: the bound to increase

    Returns:
      ub_higher: the raised bound
    """
    if ub <= -1:
      return ub / 2
    elif ub > -1 and ub < 1:
      return ub + 1
    elif ub >= 1:
      return ub * 2

  def solve_adjusted_mdp_mdptoolbox(
      self,
      state: int,
      r_array_base: np.ndarray,
      t_array: np.ndarray,
      wi_estimate: float,
      remaining_horizon: int,
  ) -> int:
    """Compute the optimal policy for the action-penalty finite horizon MDP.

    Args:
      state: the state for which we return the action of computed optimal policy
      r_array_base: the reward definition of the MDP in n_actions x n_states x
        n_states format
      t_array: the transition probabilities of the arm MDP, in n_actions x
        n_states x n_states format
      wi_estimate: the current esimate for the Whittle index
      remaining_horizon: number of time steps to construct in the finite horizon
        MDP

    Returns:
      action: the action of the optimal policy in the current round and given
      state
    """
    # Subtract the index estimate as an action penalty, along the a=1 dimension
    s1 = time.time()
    r_array = np.copy(r_array_base)
    r_array[1] = r_array_base[1] - wi_estimate
    s2 = time.time()
    # get the optimal policy of the index-adjusted Markov Decision Process
    mdp = mdptoolbox.mdp.FiniteHorizon(
        transitions=t_array,
        reward=r_array,
        discount=self.gamma,
        N=remaining_horizon,
    )
    s3 = time.time()
    mdp.run()
    s4 = time.time()
    policy = np.array(mdp.policy)  # shape (n_states, remaining_horizon)
    action = policy[state, 0]
    s5 = time.time()
    print('mdp setup time', s2-s1)
    print('object create time', s3-s2)
    print('mdp run time', s4-s3)
    print('mdp parse time', s5-s4)

    return action

  def solve_adjusted_mdp(self, state: int, r_array_base: np.ndarray,
                         t_array: np.ndarray, wi_estimate: float,
                         remaining_horizon: int) -> int:
    """Compute the optimal policy for the action-penalty finite horizon MDP.

    Args:
      state: the state for which we return the action of computed optimal policy
      r_array_base: the reward definition of the MDP in n_actions x n_states x
        n_states format
      t_array: the transition probabilities of the arm MDP, in n_actions x
        n_states x n_states format
      wi_estimate: the current esimate for the Whittle index
      remaining_horizon: number of time steps to construct in the finite horizon
        MDP

    Returns:
      action: the action of the optimal policy in the current round and given
      state
    """
    # Subtract the index estimate as an action penalty, along the a=1 dimension
    r_array = np.copy(r_array_base)
    r_array[1] = r_array_base[1] - wi_estimate

    # for this version, drop the next-state reward, since not needed
    r_array = r_array[:, :, 0]

    # get the optimal policy of the index-adjusted Markov Decision Process
    v_val, action = util.finite_horizon_value_iteration(
        remaining_horizon, t_array, r_array, state, discount=self.gamma
    )

    return v_val, action

  def compute_whittle_indexes_for_states(self, current_states: np.ndarray,
                                         remaining_horizon: int) -> np.ndarray:
    """Compute Whittle indexes for a set of arms.

    This can be parallelized over arms, eventually.

    Args:
      current_states: current state of all the arms
      remaining_horizon: number of time steps remanining in the simulation

    Returns:
      whittle_indexes: approximations for the Whittle index for each arm in
        its given state, as computed by binary search
    """
    whittle_indexes = np.zeros(self.n_arms)
    for arm in range(self.n_arms):
      whittle_indexes[arm] = self.binary_search_for_whittle_index(
          self.transition_probs[arm],
          self.rewards[arm],
          current_states[arm],
          remaining_horizon,
      )
    return whittle_indexes

  def whittle_random_tie_break(
      self, whittle_indexes_sorted, arm_inds_sorted, budget, n_arms, top_b_arms
  ):
    """Implement random tie-breaking for the Whittle index policy."""

    # only need to check for a tie after the border point, not before
    border_wi = whittle_indexes_sorted[budget-1]

    # check for edge case where we act on all arms
    if budget < n_arms and budget > 0:
      next_wi = whittle_indexes_sorted[budget]

      if (border_wi - next_wi) <= self.tie_break_epsilon:
        # compute how much budget is available to split over the tied group
        # replace with binary search or linear search to improve runtime, in
        # theory but for now, vectorized numpy ops are fast enough
        np_index_condition = (
            whittle_indexes_sorted[:budget] - border_wi
        ) > self.tie_break_epsilon
        non_tied_action_arms = arm_inds_sorted[:budget][np_index_condition]
        non_tied_budget = non_tied_action_arms.shape[0]

        remaining_budget = budget - non_tied_budget

        # act randomly over the group with the tie
        # again, could replace with binary search or linear search to improve
        #  runtime, in theory but for now, vectorized numpy ops are fast enough
        np_index_condition = (
            abs(whittle_indexes_sorted - border_wi) < self.tie_break_epsilon
        )
        tied_group = arm_inds_sorted[np_index_condition]
        tie_break_choices = np.random.choice(
            tied_group, size=remaining_budget, replace=False
        )

        top_b_arms = np.concatenate([non_tied_action_arms, tie_break_choices])

    return top_b_arms

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Returns actions according to the Whittle index policy.

    Args:
      current_states: current state of all the arms
      remaining_horizon: number of time steps remanining in the simulation

    Returns:
      actions: n_arms-length array of actions to take according to the Whittle
        index policy
    """
    if current_states is None:
      raise ValueError(
          'current_states must be provided to WhittleIndexPolicy.get_actions()')

    if remaining_horizon is None:
      err_str = 'remaining_horizon must be provided to '
      err_str += 'WhittleIndexPolicy.get_actions()'
      raise ValueError(err_str)

    effective_horizon = min(remaining_horizon, self.max_horizon)

    whittle_indexes = self.compute_whittle_indexes_for_states(
        current_states, effective_horizon)
    arm_inds_sorted = np.argsort(whittle_indexes)[::-1]
    whittle_indexes_sorted = whittle_indexes[arm_inds_sorted]

    top_b_arms = arm_inds_sorted[: self.budget]

    top_b_arms = self.whittle_random_tie_break(
        whittle_indexes_sorted,
        arm_inds_sorted,
        self.budget,
        self.n_arms,
        top_b_arms,
    )

    assert len(top_b_arms) == self.budget
    actions = np.zeros(self.n_arms, dtype=np.int32)
    actions[top_b_arms] = 1

    return actions


class NoActionPolicy(RMABPolicy):
  """Class for the No-Action RMAB policy."""

  policy_key_str = 'No Action Policy'

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return NoActionPolicy.policy_key_str

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Always return the no-action array (all zeros).

    Args:
      current_states: unused, but required in the abstract class, since
        intelligent policies will use it.
      remaining_horizon: unused, but required in the abstract class, since
        intelligent policies will use it.

    Returns:
      zero_actions: n_arms-length array of zero-actions

    """
    zero_actions = np.zeros(self.n_arms, dtype=np.int32)
    return zero_actions


class RandomHighA1cPolicy(RMABPolicy):
  """Class for implementing the A1c-based proxy policy."""

  policy_key_str = 'Random High A1c Policy'

  def __init__(self,
               n_arms: int,
               budget: int = None,
               transition_probs: np.ndarray = None,
               rewards: np.ndarray = None,
               state_index_to_tuple: dict[int, Any] = None,
               high_a1c_code: int = None,
               state_tuple_health_index: int = None,
               **kwargs) -> None:
    """Initialize the RandomHighA1cPolicy class.

    Acts randomly among arms last observed with A1c > 8.
    Designed for the DiabetesAppPartiallyObservableEnvironment.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: not required by RandomHighA1cPolicy
      rewards: not required by RandomHighA1cPolicy
      state_index_to_tuple: dict that converts state indexes to tuples
      high_a1c_code: value of the health state indicating high a1c
      state_tuple_health_index: index of the state tuple corresponding to a1c
      **kwargs: additional kwargs
    """
    super().__init__(n_arms, budget, transition_probs, rewards, **kwargs)
    self.state_dict = state_index_to_tuple
    self.high_a1c_code = high_a1c_code
    self.state_tuple_health_index = state_tuple_health_index

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return RandomHighA1cPolicy.policy_key_str

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Act randomly on arms last observed with A1c > 8.

    Args:
      current_states: current state of all arms, (s_e, s_m, s_t, s_d, s_h)
      remaining_horizon: unused, but required in the abstract class, since
        intelligent policies will use it.

    Returns:
      random_actions: n_arms-length array of random actions

    """
    actions = np.zeros(self.n_arms, dtype=np.int32)
    high_a1c_arms = []
    state_tuple_list = []
    for i, s_ind in enumerate(current_states):
      state_tuple_list.append(self.state_dict[s_ind])
      health_state_n = self.state_dict[s_ind][self.state_tuple_health_index]

      if health_state_n == self.high_a1c_code:
        high_a1c_arms.append(i)
    num_high_a1c_arms = len(high_a1c_arms)

    n_choices = min(num_high_a1c_arms, self.budget)

    choices = np.random.choice(
        a=high_a1c_arms, size=n_choices, replace=False)

    if choices.shape[0] > 0:
      actions[choices] = 1

    if n_choices < self.budget:
      other_arms = set(np.arange(self.n_arms)) - set(high_a1c_arms)
      other_arms = list(other_arms)
      other_choices = np.random.choice(
          a=other_arms, size=self.budget - n_choices, replace=False
      )

      actions[other_choices] = 1

    assert actions.sum() == self.budget

    return actions


class HighA1cRRPolicy(RMABPolicy):
  """Class for implementing the A1c-based standard-of-care proxy policy."""

  policy_key_str = 'Round Robin High A1c Policy'

  def __init__(self,
               n_arms: int,
               budget: int = None,
               transition_probs: np.ndarray = None,
               rewards: np.ndarray = None,
               state_index_to_tuple: dict[int, Any] = None,
               high_a1c_code: int = None,
               state_tuple_health_index: int = None,
               horizon: int = None,
               **kwargs
               ) -> None:
    """Initialize the RandomHighA1cPolicy class.

    Contains code for acting randomly among arms
    last observed with A1c > 8.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: not required by RandomHighA1cPolicy
      rewards: not required by RandomHighA1cPolicy
      state_index_to_tuple: dict that converts state indexes to tuples
      high_a1c_code: value of the health state indicating high a1c
      state_tuple_health_index: index of the state tuple corresponding to a1c
      horizon: length of simulation
      **kwargs: additional kwargs
    """
    super().__init__(n_arms, budget, transition_probs, rewards, **kwargs)
    self.state_dict = state_index_to_tuple
    self.high_a1c_code = high_a1c_code
    self.state_tuple_health_index = state_tuple_health_index
    self.round_last_acted = np.zeros(n_arms, dtype=int)
    self.horizon = horizon

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return HighA1cRRPolicy.policy_key_str

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Act randomly on arms last observed with A1c > 8.

    Args:
      current_states: current state of all arms, (s_e, s_m, s_t, s_d, s_h)
      remaining_horizon: unused, but required in the abstract class, since
        intelligent policies will use it.

    Returns:
      random_actions: n_arms-length array of random actions

    """
    t = self.horizon - remaining_horizon

    rounds_since_acting = t - self.round_last_acted

    actions = np.zeros(self.n_arms, dtype=np.int32)
    high_a1c_arms = []
    state_tuple_list = []
    for i, s_ind in enumerate(current_states):
      state_tuple_list.append(self.state_dict[s_ind])
      health_state_n = self.state_dict[s_ind][self.state_tuple_health_index]

      if health_state_n == self.high_a1c_code:
        high_a1c_arms.append(i)

    low_a1c_arms = list(set(np.arange(self.n_arms)) - set(high_a1c_arms))

    num_high_a1c_arms = len(high_a1c_arms)
    n_choices = min(num_high_a1c_arms, self.budget)

    # define out the lowa1c arms
    rounds_since_acting[low_a1c_arms] = -1

    sorted_high_a1c_arms = np.argsort(rounds_since_acting)[::-1]

    choices = sorted_high_a1c_arms[:n_choices]

    if choices.shape[0] > 0:
      actions[choices] = 1

    other_choices = None
    if n_choices < self.budget:
      rounds_since_acting = t - self.round_last_acted
      rounds_since_acting[high_a1c_arms] = -1  # now define out the higha1c arms
      sorted_low_a1c_arms = np.argsort(rounds_since_acting)[::-1]
      other_choices = sorted_low_a1c_arms[:self.budget - n_choices]

      actions[other_choices] = 1

    self.round_last_acted[choices] = 0
    self.round_last_acted[other_choices] = 0

    assert actions.sum() == self.budget

    return actions


class EquitableLPPolicyMMR(WhittleIndexPolicy):
  """Implements the Equitable lp Policy."""

  # TODO(jakillian) allow passing proxy_transition_probs used for planning
  # (shorter horizons)

  policy_key_str = 'Equitable LP Policy: MMR'

  def __init__(self,
               n_arms: int,
               budget: int,
               transition_probs: np.ndarray,
               rewards: np.ndarray,
               horizon: int = None,
               lambda_lb: float = 0,
               lambda_ub: float = None,
               index_lb: float = -1,
               index_ub: float = 1,
               gamma: float = 0.9999,
               binary_search_tolerance: float = 1e-1,
               group_map: list[int] = None,
               subsample_size: int = 0,
               regularized_version: bool = False,
               active_arms_helper: object = None,
               max_horizon: int = 20,
               **kwargs,
               ) -> None:
    """Initialize the EquitableLPPolicy class.

    Contains code for computing the
    Equitable lp Policy for a given RMAB.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      horizon: length of simulation
      lambda_lb: lower bound for solution value of lambda
      lambda_ub: upper bound for solution value of lambda
      index_lb: lower bound of for whittle index binary search
      index_ub: upper bound of for whittle index binary search
      gamma: discount factor rewards
      binary_search_tolerance: tolerance for binary search of whittle index
      group_map: n_arms-length list defining the groups of all arms
      subsample_size: (currently unused), number of arms to subsample to
        to estimate budgets on a smaller problem.
      regularized_version: whether or not to include the lambda term in the
        objective function
      active_arms_helper: helper function to determine which arms are active
        by interpreting their state codes.
      **kwargs: additional kwargs.
    """
    super().__init__(
        n_arms,
        budget=budget,
        transition_probs=transition_probs,
        rewards=rewards,
        index_lb=index_lb,
        index_ub=index_ub,
        gamma=gamma,
        binary_search_tolerance=binary_search_tolerance,
        max_horizon=max_horizon,
        **kwargs,
        )
    self.horizon = horizon
    self.lambda_lb = lambda_lb
    if lambda_ub is None:
      self.lambda_ub = rewards.max()/(1-gamma)
    else:
      self.labmda_ub = lambda_ub
    self.gamma = gamma
    self.group_map = group_map
    self.subsample_size = subsample_size
    self.unique_groups = sorted(list(set(group_map)))
    self.n_actions = 2  # only binary action implemented currently
    self.cost_vector = [0, 1]
    self.n_states = transition_probs.shape[1]
    self.regularized_version = regularized_version

    self.active_arms_helper = active_arms_helper

    self.base_model_path = util.Config.BASE_LP_MODEL_PATH
    self.max_horizon = max_horizon

    # self.lp_setup()

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return EquitableLPPolicyMMR.policy_key_str

  def get_active_arms(self, states):
    return self.active_arms_helper.get_active_arms(states)

  # Prospective implementation. Helper function for multi-action.
  def lp_setup(self):
    """Create a PuLP LP for solving the Lagrange relaxation."""

    # TODO(jakillian): uncomment and separate this class
    # assert self.subsample_size >= len(self.unique_groups)
    # assert self.subsample_size <= self.n_arms

    # compute how many arms from each group will go into the subsample
    # ensure there is at least one per group --
    # ok if this goes slightly over subsample_size
    self.num_sampled_arms_per_group = np.zeros(
        len(self.unique_groups), dtype=int
    )
    for i, group in enumerate(self.unique_groups):
      arms_in_group = (self.group_map == group).sum()
      self.num_sampled_arms_per_group[i] = int(
          arms_in_group / self.n_arms * self.subsample_size
      )
      if self.num_sampled_arms_per_group[i] < 1:
        self.num_sampled_arms_per_group[i] = 1

    self.lp_resources = {}
    for i, group in enumerate(self.unique_groups):
      group_size = self.num_sampled_arms_per_group[i]
      dummy_t = self.transition_probs[:group_size]
      dummy_r = self.rewards[:group_size]
      dummy_state = np.zeros(group_size, dtype=np.int32)
      dummy_timestep = 0
      n_arms = dummy_t.shape[0]
      n_states = dummy_t.shape[1]
      n_actions = 2

      filename = os.path.join(
          self.base_model_path,
          'base_model_n%i_s%i_a%i_h%i.json'
          % (n_arms, n_states, n_actions, self.horizon),
      )
      if os.path.exists(filename):
        self.lp_resources[group] = self.lp_load_from_edited_dict(
            filename, self.horizon, dummy_t, dummy_r, dummy_timestep
        )
      else:
        ret_data = mathprog_methods.lagrange_relaxation_finite_horizon(
            dummy_t,
            dummy_r,
            self.cost_vector,
            self.budget,
            self.horizon,
            dummy_timestep,
            dummy_state,
            lambda_lim=self.lambda_ub,
            gamma=self.gamma,
        )
        # v_values = ret_data[0]
        # lambda_star = ret_data[1]
        # objective_value = ret_data[2]
        group_lp_resources = ret_data[3]
        # save this model as a template for future loading
        group_lp_resources['lp'].to_json(filename)
        self.lp_resources[group] = group_lp_resources

  # Prospective implementation. Helper function for multi-action.
  def edit_lp_constraints_finite_horizon(
      self, t_probs, r, c, h, timestep, lp_dict
  ):
    """Edit constraints of a PuLP LP directly, for computational gains."""

    n_procs = t_probs.shape[0]
    n_states = t_probs.shape[1]
    n_actions = t_probs.shape[2]

    ind = 0
    for n in range(n_procs):
      for s in range(n_states):
        for a in range(n_actions):
          # keep order the same as when created for parsing efficiency
          for t in range(h+1):
            if t >= timestep and t < h:  # but only change what's necessary
              constraint_name = 'L_n%i_s%i_a%i_t%s'%(n, s, a, t)
              # try:
              constraint_dict = lp_dict['constraints'][ind]
              # except:
              #   import pdb; pdb.set_trace()
              if constraint_dict['name'] != constraint_name: raise ValueError
              if constraint_dict['sense'] != 1: raise ValueError
              constraint_dict['constant'] = -r[n, s]

              # each constraint will have n_states L variables at the t+1
              # timestep, one L variable at the t timestep and s state,
              # and 1 index variable
              for coef in constraint_dict['coefficients']:
                if coef['name'] == 'index':
                  coef['value'] = c[a]
                else:
                  # L variable names are L_proc_state_timestep
                  # so need second-to-last index for state
                  # and last index for time
                  l_var_name_inds = coef['name'].split('_')
                  sp = int(l_var_name_inds[-2])
                  l_timestep = int(l_var_name_inds[-1])
                  coef['value'] = -t_probs[n, s, a, sp] * self.gamma
                  if l_timestep == t:
                    coef['value'] = 1

            ind += 1

  # Prospective implementation. Helper function for multi-action.
  def lp_load_from_edited_dict(
      self, filename, horizon, t_probs, r, timestep, debugging=False
  ):
    """Load PuLP LP from dict for computational gains."""
    # load base file
    lp_data = ujson.load(open(filename, 'r'))
    # run customized function to edit pulp dictionary representations
    # of the lp (fast)
    self.edit_lp_constraints_finite_horizon(
        t_probs, r, self.cost_vector, horizon, timestep, lp_data
    )

    # PuLP forces you to load from file, so use tempfiles to do
    # this safely and without running up disk space
    tmp_fd, tmp_path = tempfile.mkstemp()
    try:
      with os.fdopen(tmp_fd, 'w') as tmp:
        ujson.dump(lp_data, tmp)

      variables, lp = pulp.LpProblem.from_json(tmp_path)

      if debugging:
        lp.to_json(filename+'.edited_copy')

      index_variable = variables['index']
      del variables['index']  # this is the only other variable, rest are L_vars

      lp_resources = {
          'lp': lp,
          'L': variables,
          'index_variable': index_variable,
      }

    finally:
      os.remove(tmp_path)

    return lp_resources

  # Prospective implementation. Use for multi-action.
  def solve_group_lp(
      self, group, group_budget, group_states, group_inds, timestep
  ):
    """Solve a the Largrange relaxation for a group."""
    t_probs = self.transition_probs[group_inds]
    r = self.rewards[group_inds]
    v_values, lambda_star, objective_value, group_lp_resources = (
        mathprog_methods.lagrange_relaxation_finite_horizon(
            t_probs,
            r,
            self.cost_vector,
            group_budget,
            self.horizon,
            timestep,
            group_states,
            lambda_lim=self.lambda_ub,
            gamma=self.gamma,
            lp_resources=self.lp_resources[group],
        )
    )
    self.lp_resources[group] = group_lp_resources
    group_avg_objective = (
        objective_value / self.num_sampled_arms_per_group[group]
    )

    return group_avg_objective, v_values, lambda_star

  # Prospective implementation. Use for multi-action and larger problem sizes
  def get_equitable_budgets_sample(self, states, timestep):
    """Compute MMR using subsample of arms for computational gains."""
    # Choose a subsample of arms per group to do the budget allocation
    sampled_group_inds = {group: [] for group in self.unique_groups}
    for i, group in enumerate(self.unique_groups):
      choices = np.arange(self.n_arms)[self.group_map == group]
      sampled_group_inds[group] = np.random.choice(
          choices, size=self.num_sampled_arms_per_group[i], replace=False
      )

      # once arms are picked, load in the transition probs to each
      # lp in lp_resources
      t_probs = self.transition_probs[sampled_group_inds[group]]
      r = self.rewards[sampled_group_inds[group]]
      n_arms = t_probs.shape[0]
      n_states = t_probs.shape[1]
      n_actions = 2

      filename = os.path.join(
          self.base_model_path,
          'base_model_n%i_s%i_a%i_h%i.json'
          % (n_arms, n_states, n_actions, self.horizon),
      )
      if os.path.exists(filename):
        self.lp_resources[group] = self.lp_load_from_edited_dict(
            filename, self.horizon, t_probs, r, timestep
        )
      else:
        raise ValueError('lp base model should exist, but could not locate.')

    # compute a smaller, proportional budget
    b_sub = int(max(1, self.budget / self.n_arms * self.subsample_size))

    # Now start to compute equitable budget allocation -- first setup structures
    group_budgets = np.zeros(len(self.unique_groups))
    group_states = {
        group: states[sampled_group_inds[group]] for group in self.unique_groups
    }
    group_v_values = {group: None for group in self.unique_groups}
    group_lambda_stars = {group: None for group in self.unique_groups}

    # start by solving each group with 0 budget
    group_avg_objectives = np.zeros(len(self.unique_groups))
    for group in self.unique_groups:
      group_inds = sampled_group_inds[group]
      ret_data = self.solve_group_lp(
          group,
          group_budgets[group],
          group_states[group],
          group_inds,
          timestep
      )
      group_avg_objectives[group] = ret_data[0]
      group_v_values[group] = ret_data[1]
      group_lambda_stars[group] = ret_data[2]

    # Now do water filling to compute equitable budget split
    print(self.num_sampled_arms_per_group)
    print(sampled_group_inds)
    print(group_avg_objectives)
    print('starting water filling')
    full_budget_groups = group_budgets >= self.num_sampled_arms_per_group
    for _ in range(b_sub):
      next_group_check = np.copy(group_avg_objectives)
      # do this so we don't allocate more budget to one group than arms in group
      next_group_check[full_budget_groups] = np.inf
      min_group = np.argmin(next_group_check)
      group_budgets[min_group] += 1
      group_inds = sampled_group_inds[min_group]
      ret_data = self.solve_group_lp(
          min_group,
          group_budgets[min_group],
          group_states[min_group],
          group_inds,
          timestep
      )
      group_avg_objectives[min_group] = ret_data[0]
      group_v_values[min_group] = ret_data[1]
      group_lambda_stars[min_group] = ret_data[2]
      print('budgets', group_budgets)
      print('objectives', group_avg_objectives)
      full_budget_groups = group_budgets >= self.num_sampled_arms_per_group

    # now do some quick rounding to scale group budgets back to size,
    # randomly assigning remainders
    scaled_group_budgets = group_budgets / self.subsample_size * self.n_arms
    scaled_group_budgets = scaled_group_budgets.astype(int)
    group_sizes = [
        (self.group_map == group).sum() for group in self.unique_groups
    ]
    while scaled_group_budgets.sum() < self.budget:
      remaining_budget_groups = np.arange(len(self.unique_groups))[
          scaled_group_budgets < group_sizes
      ]
      group_to_add_to_next = np.random.choice(remaining_budget_groups)
      scaled_group_budgets[group_to_add_to_next] += 1

    return scaled_group_budgets

  def convert_and_compute_v(
      self, current_states, lambda_value, t_probs, r, arm, remaining_horizon
  ):
    """Compute the value function with value iteration."""
    # Change from S,A,S format to A,S,S format
    t_array = np.swapaxes(t_probs[arm], 0, 1)
    # Rewards also need to be in A,S,S format
    # (the rest of the repo assumes rewards are only function of current state)
    r_array_base = np.zeros(t_array.shape)
    n_states = t_array.shape[1]
    for a in range(self.n_actions):
      for s in range(n_states):
        r_array_base[a, :, s] += r[arm]

    v_arm, _ = self.solve_adjusted_mdp(
        current_states[arm],
        r_array_base,
        t_array,
        lambda_value,
        remaining_horizon,
    )
    return v_arm

  def get_equitable_budgets_whittle(self, states, remaining_horizon):
    """Implements the MMR equitable budget policy using whittle indexes.

    Args:
      states: current states of all arms
      remaining_horizon: remaining time in the simulation

    Returns:
      1: per-group budgets, using MMR logic
      2: whittle indexes for each arm, grouped by group

    """

    # start by computing the whittle indexes of each group
    # then find each group's cutoff point given the budget.
    # Use that cutoff to compute the lambda and subsequent value functions.
    # Then iterate.

    # get the list of active arms
    active = self.get_active_arms(states)

    # Now start to compute equitable budget allocation -- first setup structures
    group_budgets = np.zeros(len(self.unique_groups), dtype=np.int32)
    group_inds_dict = {
        group: self.group_map == group for group in self.unique_groups
    }
    group_sizes = [group_inds_dict[group].sum() for group in self.unique_groups]
    group_sizes_active = [
        (group_inds_dict[group] & active).sum() for group in self.unique_groups
    ]
    group_whittles = {
        group: np.zeros(group_sizes[group]) for group in self.unique_groups
    }
    group_whittles_sorted = {
        group: np.zeros(group_sizes[group]) for group in self.unique_groups
    }
    group_states = {
        group: states[(self.group_map == group)] for group in self.unique_groups
    }
    # group_v_values = {group: None for group in self.unique_groups}
    # group_lambda_stars = {group: None for group in self.unique_groups}

    # start by solving for the whittle indexes of each group
    # and instantiate the value functions by solving for each
    # assuming the no-budget MDP, which is the same as
    # the MDP with very high cost action
    group_avg_objectives = np.zeros(len(self.unique_groups))
    for group in self.unique_groups:
      group_inds = group_inds_dict[group]
      original_arm_inds = np.arange(self.n_arms)[group_inds]
      group_size = group_sizes[group]
      group_size_active = group_sizes_active[group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[group]
      v_group = 0
      for arm in range(group_size):
        group_whittles[group][arm] = self.binary_search_for_whittle_index(
            t_probs[arm],
            r[arm],
            current_states[arm],
            remaining_horizon,
        )

        lambda_value = 1e6
        # Compute value function if this arm is active
        if active[original_arm_inds[arm]]:
          v_group += self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )

      group_avg_objectives[group] = (
          v_group / group_size_active
      )  # first round assume no budget
      if group_size_active == 0:
        # ensure no budget is allocated for MMR, by assigning inf
        group_avg_objectives[group] = np.inf

      arm_inds_sorted = np.argsort(group_whittles[group])[::-1]
      group_whittles_sorted[group] = group_whittles[group][arm_inds_sorted]

    # Now carry out the water filling algorithm, assigning additional unit
    # budget to each group with the smallest average value function, checking
    # edge cases.
    # print('begin whittle water filling up to', self.budget)
    # print(group_avg_objectives)
    full_budget_groups = group_budgets >= group_sizes_active
    for _ in range(self.budget):
      s1 = time.time()
      next_group_check = np.copy(group_avg_objectives)

      # assign inf to full-budget groups so they are not considered for
      # more budget by MMR
      next_group_check[full_budget_groups] = np.inf

      # check for all inf here and break if so
      if (next_group_check == np.inf).all():
        # print('all groups full')
        break

      # Find min group and assign budget
      min_group = np.argmin(next_group_check)
      group_budgets[min_group] += 1

      # Recompute the average value function over the group with
      # new lambda_value
      group_inds = group_inds_dict[min_group]
      group_size = group_sizes[min_group]
      group_size_active = group_sizes_active[min_group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[min_group]

      original_arm_inds = np.arange(self.n_arms)[group_inds]

      # set lambda as the value between the last whittle index and the next
      gb = group_budgets[min_group]
      if gb < group_size_active:
        lambda_value = group_whittles_sorted[min_group][gb-1: gb+1].mean()
      else:
        lambda_value = group_whittles_sorted[min_group][gb-1]

      # Recompute value functions of active arms
      v_group = 0
      for arm in range(group_size):
        if active[original_arm_inds[arm]]:
          v_group += self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )

      if not self.regularized_version:
        group_avg_objectives[min_group] = (
            v_group
            + lambda_value
            * sum([(gb) * (self.gamma**t) for t in range(remaining_horizon)])
        ) / group_size_active
      else:
        group_avg_objectives[min_group] = (
            v_group
            + sum([(gb) * (self.gamma**t) for t in range(remaining_horizon)])
        ) / group_size_active

      # print('budgets', group_budgets)
      # print('objectives', group_avg_objectives)
      full_budget_groups = group_budgets >= group_sizes_active
      s2 = time.time()
      # print('One round water filling took %ss'%(s2-s1))

    # Finally ranndomly allocate remaining budget if not enough active arms
    if (next_group_check == np.inf).all():
      unique_groups_np = np.array(self.unique_groups)
      # recompute group sizes with all arms
      group_sizes = [
          (self.group_map == group).sum() for group in self.unique_groups
      ]
      b_used = group_budgets.sum()
      under_budget_groups = group_budgets < group_sizes
      for _ in range(b_used, self.budget):
        random_group = np.random.choice(unique_groups_np[under_budget_groups])
        group_budgets[random_group] += 1
        under_budget_groups = group_budgets < group_sizes

    # print('group_sizes', group_sizes)
    # print('group_sizes_active', group_sizes_active)
    # print('group budgets', group_budgets)
    # print('active', active)

    return group_budgets, group_whittles

  def group_wip(
      self, group_budget, group, whittles=None
  ):
    """Act on each group according to their budget and Whittle index policy."""
    group_inds = self.group_map == group
    group_size = group_inds.sum()
    # t_probs = self.transition_probs[group_inds]
    # r = self.rewards[group_inds]
    # current_states = states[group_inds]

    # If pre-computed Whittle indexes are not passed in
    # if whittles is None:
    #   whittle_indexes = np.zeros(group_size)
    #   for arm in range(group_size):
    #     whittle_indexes[arm] = self.binary_search_for_whittle_index(
    #         t_probs[arm],
    #         r[arm],
    #         current_states[arm],
    #         remaining_horizon,
    #     )
    # else:
    whittle_indexes = whittles[group]

    arm_inds_sorted = np.argsort(whittle_indexes)[::-1]
    whittle_indexes_sorted = whittle_indexes[arm_inds_sorted]

    top_b_arms = arm_inds_sorted[:group_budget]

    top_b_arms = self.whittle_random_tie_break(
        whittle_indexes_sorted,
        arm_inds_sorted,
        group_budget,
        group_size,
        top_b_arms,
    )

    actions = np.zeros(group_size, dtype=np.int32)
    actions[top_b_arms] = 1
    # print(actions)
    return actions

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Returns actions according to the Equitable lp policy.

    Args:
      current_states: current state of all the arms
      remaining_horizon: number of time steps remanining in the simulation

    Returns:
      actions: n_arms-length array of actions to take according to the Whittle
        index policy
    """
    if current_states is None:
      raise ValueError(
          'current_states must be provided to EquitableLPPolicy.get_actions()'
      )

    if remaining_horizon is None:
      err_str = 'remaining_horizon must be provided to '
      err_str += 'EquitableLPPolicy.get_actions()'
      raise ValueError(
          err_str
      )

    # timestep = self.horizon - remaining_horizon

    print('state', current_states)
    effective_horizon = min(remaining_horizon, self.max_horizon)

    group_budgets, whittles = self.get_equitable_budgets_whittle(
        current_states, effective_horizon
    )

    # get actions for each group according to the Whittle index policy,
    # with the given group budgets
    all_actions = np.zeros(self.n_arms, dtype=np.int32)
    for group in self.unique_groups:
      group_inds = self.group_map == group
      all_actions[group_inds] = self.group_wip(
          group_budgets[group],
          group,
          whittles,
      )

    print('Done')
    print(all_actions)
    assert all_actions.sum() == self.budget

    return all_actions


class EquitableLPPolicyMNW(EquitableLPPolicyMMR):
  """Implements the Equitable lp Policy for MNW objective."""

  policy_key_str = 'Equitable LP Policy: MNW'

  def __init__(self,
               n_arms: int,
               budget: int,
               transition_probs: np.ndarray,
               rewards: np.ndarray,
               horizon: int = None,
               lambda_lb: float = 0,
               lambda_ub: float = None,
               index_lb: float = -1,
               index_ub: float = 1,
               gamma: float = 0.9999,
               binary_search_tolerance: float = 1e-1,
               group_map: list[int] = None,
               subsample_size: int = 0,
               active_arms_helper: object = None,
               **kwargs,
               ) -> None:
    """Initialize the EquitableLPPolicy class.

    Contains code for computing the
    Equitable lp Policy for a given RMAB.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      horizon: length of simulation
      lambda_lb: lower bound for solution value of lambda
      lambda_ub: upper bound for solution value of lambda
      index_lb: lower bound of for whittle index binary search
      index_ub: upper bound of for whittle index binary search
      gamma: discount factor rewards
      binary_search_tolerance: tolerance for binary search of whittle index
      group_map: n_arms-length list defining the groups of all arms
      subsample_size: (currently unused), number of arms to subsample to
        to estimate budgets on a smaller problem.
      active_arms_helper: helper function to determine which arms are active
        by interpreting their state codes.
      **kwargs: additional kwargs.
    """
    super().__init__(
        n_arms,
        budget=budget,
        transition_probs=transition_probs,
        rewards=rewards,
        horizon=horizon,
        index_lb=index_lb,
        index_ub=index_ub,
        gamma=gamma,
        binary_search_tolerance=binary_search_tolerance,
        lambda_lb=lambda_lb,
        lambda_ub=lambda_ub,
        group_map=group_map,
        subsample_size=subsample_size,
        active_arms_helper=active_arms_helper,
        **kwargs
        )

    self.unique_groups = sorted(list(set(group_map)))
    self.n_actions = 2  # only binary action implemented currently
    self.cost_vector = [0, 1]
    self.n_states = transition_probs.shape[1]

    self.base_model_path = util.Config.BASE_LP_MODEL_PATH

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return EquitableLPPolicyMNW.policy_key_str

  def get_equitable_budgets_whittle(self, states, remaining_horizon):
    """Implements the MNW equitable budget policy using whittle indexes.

    Args:
      states: current states of all arms
      remaining_horizon: remaining time in the simulation

    Returns:
      1: per-group budgets, using MNW logic
      2: whittle indexes for each arm, grouped by group

    """

    # start by computing the whittle indexes of each group
    # then find each group's cutoff point given the budget
    # use that cutoff to compute the lambda and subsequent value functions
    # iterate

    # get the list of active arms
    active = self.get_active_arms(states)

    # Now start to compute equitable budget allocation -- first setup structures
    group_budgets = np.zeros(len(self.unique_groups), dtype=np.int32)
    group_inds_dict = {
        group: self.group_map == group for group in self.unique_groups
    }
    group_sizes = [
        group_inds_dict[group].sum() for group in self.unique_groups
    ]
    group_sizes_active = [
        (group_inds_dict[group] & active).sum() for group in self.unique_groups
    ]
    group_whittles = {
        group: np.zeros(group_sizes[group]) for group in self.unique_groups
    }
    group_whittles_sorted = {
        group: np.zeros(group_sizes[group]) for group in self.unique_groups
    }
    group_states = {
        group: states[(self.group_map == group)] for group in self.unique_groups
    }
    # group_v_values = {group: None for group in self.unique_groups}
    # group_lambda_stars = {group: None for group in self.unique_groups}

    # start by solving for the whittle indexes of each group
    # and instantiate the value functions by solving for each
    # assuming the no-budget MDP, which is the same as
    # the MDP with very high cost action
    group_avg_b0 = np.zeros(len(self.unique_groups))
    group_avg_b1 = np.zeros(len(self.unique_groups))
    group_avg_log_deltas = np.zeros(len(self.unique_groups))
    for group in self.unique_groups:
      group_inds = group_inds_dict[group]
      original_arm_inds = np.arange(self.n_arms)[group_inds]
      group_size = group_sizes[group]
      group_size_active = group_sizes_active[group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[group]
      v_group = 0
      for arm in range(group_size):
        group_whittles[group][arm] = self.binary_search_for_whittle_index(
            t_probs[arm],
            r[arm],
            current_states[arm],
            remaining_horizon,
        )

        lambda_value = 1e6
        # Compute value function if this arm is active
        if active[original_arm_inds[arm]]:
          v_group += self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )

      group_avg_b0[group] = v_group  # first round has no budget

      # if no active arms, assign 0 for b0 and b1 so MNW never considers
      if group_size_active == 0:
        group_avg_b0[group] = 0

      arm_inds_sorted = np.argsort(group_whittles[group])[::-1]
      group_whittles_sorted[group] = group_whittles[group][arm_inds_sorted]

    # now that whittle indexes are computed, can compute initial deltas
    # init b1 and b0 to same, so undersized groups give -np.inf on log
    group_avg_b1[:] = group_avg_b0[:]
    under_budget_groups = np.arange(len(self.unique_groups))[
        group_budgets < group_sizes_active
    ]
    for group in under_budget_groups:
      next_b = group_budgets[group] + 1
      if next_b < group_size_active:
        lambda_value = group_whittles_sorted[group][next_b-1: next_b+1].mean()
      else:
        lambda_value = group_whittles_sorted[group][next_b-1]

      group_inds = group_inds_dict[group]
      group_size = group_sizes[group]
      group_size_active = group_sizes_active[group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[group]

      original_arm_inds = np.arange(self.n_arms)[group_inds]
      v_group = 0
      for arm in range(group_size):
        # Compute value function if this arm is active
        if active[original_arm_inds[arm]]:
          v_group += self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )

      vsum = sum(
          [(next_b) * (self.gamma**t) for t in range(remaining_horizon)]
      )
      group_avg_b1[group] = v_group + lambda_value * vsum

    # now compute log deltas
    # group_avg_log_deltas = np.log(group_avg_b1 - group_avg_b0)
    group_avg_log_deltas = (np.log(group_avg_b1) - np.log(group_avg_b0))

    print('begin greedy selection up to', self.budget)
    print(group_avg_b0)
    print(group_avg_b1)
    print(group_avg_log_deltas)
    full_budget_groups = group_budgets >= group_sizes_active
    next_group_check = np.copy(group_avg_log_deltas)
    # so we don't allocate more budget to one group than arms in group
    next_group_check[full_budget_groups] = -np.inf
    for _ in range(self.budget):

      s1 = time.time()
      next_group_check = np.copy(group_avg_log_deltas)
      # so we don't allocate more budget to one group than arms in group
      next_group_check[full_budget_groups] = -np.inf
      # check for all inf here and break if so
      if (next_group_check == -np.inf).all():
        print('all groups full')
        break
      mx_grp = np.argmax(next_group_check)
      group_budgets[mx_grp] += 1

      # only compute next value function if we have budget left
      if group_budgets[mx_grp] >= group_sizes_active[mx_grp]:
        group_avg_b0[mx_grp] = group_avg_b1[mx_grp]
      else:
        group_inds = group_inds_dict[mx_grp]
        group_size = group_sizes[mx_grp]
        group_size_active = group_sizes_active[mx_grp]
        t_probs = self.transition_probs[group_inds]
        r = self.rewards[group_inds]
        current_states = group_states[mx_grp]

        original_arm_inds = np.arange(self.n_arms)[group_inds]

        # set lambda as the value between the last whittle index and the next
        next_b = group_budgets[mx_grp] + 1
        if next_b < group_size_active:
          lambda_value = group_whittles_sorted[mx_grp][next_b-1: next_b+1].mean()
        else:
          lambda_value = group_whittles_sorted[mx_grp][next_b-1]

        v_group = 0
        for arm in range(group_size):
          # Compute value function if this arm is active
          if active[original_arm_inds[arm]]:
            v_group += self.convert_and_compute_v(
                current_states, lambda_value, t_probs, r, arm, remaining_horizon
            )

        group_avg_b0[mx_grp] = group_avg_b1[mx_grp]

        h = remaining_horizon
        vsum = sum([(next_b)*(self.gamma**t) for t in range(h)])
        group_avg_b1[mx_grp] = v_group + lambda_value*vsum

      gap = (np.log(group_avg_b1[mx_grp]) - np.log(group_avg_b0[mx_grp]))
      group_avg_log_deltas[mx_grp] = gap

      print('budgets', group_budgets)
      print('objectives', group_avg_log_deltas)
      print(group_avg_b0)
      print(group_avg_b1)
      print(group_avg_b1/group_avg_b0)
      print(group_avg_log_deltas)
      full_budget_groups = group_budgets >= group_sizes_active
      s2 = time.time()
      print('One round water filling took %ss'%(s2-s1))
      print()

    # ranndomly allocate remaining budget
    if (next_group_check == -np.inf).all():
      unique_groups_np = np.array(self.unique_groups)
      # recompute group sizes without
      group_sizes = [
          (self.group_map == group).sum() for group in self.unique_groups
      ]
      b_used = group_budgets.sum()
      under_budget_groups = group_budgets < group_sizes
      for _ in range(b_used, self.budget):
        random_group = np.random.choice(unique_groups_np[under_budget_groups])
        group_budgets[random_group] += 1
        under_budget_groups = group_budgets < group_sizes

    print('group_sizes', group_sizes)
    print('group_sizes_active', group_sizes_active)
    print('group budgets', group_budgets)
    print('active', active)

    return group_budgets, group_whittles


class EquitableLPPolicyMNWEqualGroups(EquitableLPPolicyMMR):
  """Implements the Equitable LP Policy for MNW objective
     with group balacing."""

  policy_key_str = 'Equitable LP Policy: MNW Equal Groups'

  def __init__(self,
               n_arms: int,
               budget: int,
               transition_probs: np.ndarray,
               rewards: np.ndarray,
               horizon: int = None,
               lambda_lb: float = 0,
               lambda_ub: float = None,
               index_lb: float = -1,
               index_ub: float = 1,
               gamma: float = 0.9999,
               binary_search_tolerance: float = 1e-1,
               group_map: list[int] = None,
               subsample_size: int = 0,
               active_arms_helper: object = None,
               **kwargs,
               ) -> None:
    """Initialize the EquitableLPPolicyMNWEqualGroups class.

    Contains code for computing the
    Equitable LP Policy, using the MNW group equalization approach.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      horizon: length of simulation
      lambda_lb: lower bound for solution value of lambda
      lambda_ub: upper bound for solution value of lambda
      index_lb: lower bound of for whittle index binary search
      index_ub: upper bound of for whittle index binary search
      gamma: discount factor rewards
      binary_search_tolerance: tolerance for binary search of whittle index
      group_map: n_arms-length list defining the groups of all arms
      subsample_size: (currently unused), number of arms to subsample to
        to estimate budgets on a smaller problem.
      active_arms_helper: helper function to determine which arms are active
        by interpreting their state codes.
      **kwargs: additional kwargs.
    """
    super().__init__(
        n_arms,
        budget=budget,
        transition_probs=transition_probs,
        rewards=rewards,
        horizon=horizon,
        index_lb=index_lb,
        index_ub=index_ub,
        gamma=gamma,
        binary_search_tolerance=binary_search_tolerance,
        lambda_lb=lambda_lb,
        lambda_ub=lambda_ub,
        group_map=group_map,
        subsample_size=subsample_size,
        active_arms_helper=active_arms_helper,
        **kwargs
        )

    self.unique_groups = sorted(list(set(group_map)))
    self.n_actions = 2  # only binary action implemented currently
    self.cost_vector = [0, 1]
    self.n_states = transition_probs.shape[1]

    self.base_model_path = util.Config.BASE_LP_MODEL_PATH

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return EquitableLPPolicyMNWEqualGroups.policy_key_str

  def get_equitable_budgets_whittle(self, states, remaining_horizon):
    """Implements the MNW equitable budget policy using whittle indexes
       and group balancing.

    Args:
      states: current states of all arms
      remaining_horizon: remaining time in the simulation

    Returns:
      1: per-group budgets, using MNW logic
      2: whittle indexes for each arm, grouped by group

    """

    # start by computing the whittle indexes of each group
    # then find each group's cutoff point given the budget
    # use that cutoff to compute the lambda and subsequent value functions
    # iterate

    # get the list of active arms
    active = self.get_active_arms(states).astype(bool)

    # Now start to compute equitable budget allocation -- first setup structures
    group_budgets = np.zeros(len(self.unique_groups), dtype=np.int32)
    group_inds_dict = {
        group: self.group_map == group for group in self.unique_groups
    }
    group_sizes = [
        group_inds_dict[group].sum() for group in self.unique_groups
    ]
    group_sizes_active = [
        (group_inds_dict[group] & active).sum() for group in self.unique_groups
    ]
    group_whittles = {
        group: np.zeros(group_sizes[group]) for group in self.unique_groups
    }
    group_states = {
        group: states[(self.group_map == group)] for group in self.unique_groups
    }

    # Select a random list of dummy arms to use to
    # simulate equalizing the group size later
    # Key idea: Use the dummies to figure out the budget *ratio* each group
    #   should get when we have controlled for sizes... then return those ratios
    #   rescaled by the true group sizes.
    group_dummy_arms = {}
    max_group_size = max(group_sizes_active)

    for group in self.unique_groups:
      size_difference = max_group_size - group_sizes_active[group]
      active_group_inds = group_inds_dict[group] & active
      active_group_arms = np.arange(self.n_arms)[active_group_inds]
      ## if no active arms, take first. Budget will rescale to 0.
      if active_group_arms.shape[0] < 1:
        active_group_arms = group_inds_dict[group][:1]
      
      group_dummy_arms[group] = np.random.choice(
          active_group_arms,
          size=size_difference,
          replace=True,
      )

    # Need dummy copies of whittles internally too, but order doesn't matter
    group_whittles_sorted = {
        group: [] for group in self.unique_groups
    }
    

    # start by solving for the whittle indexes of each group
    # and instantiate the value functions by solving for each
    # assuming the no-budget MDP, which is the same as
    # the MDP with very high cost action
    group_avg_b0 = np.zeros(len(self.unique_groups))
    group_avg_b1 = np.zeros(len(self.unique_groups))
    group_avg_log_deltas = np.zeros(len(self.unique_groups))
    for group in self.unique_groups:
      group_inds = group_inds_dict[group]
      original_arm_inds = np.arange(self.n_arms)[group_inds]
      group_size = group_sizes[group]
      group_size_active = group_sizes_active[group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[group]
      v_group = 0
      for arm in range(group_size):
        whittle_index = self.binary_search_for_whittle_index(
            t_probs[arm],
            r[arm],
            current_states[arm],
            remaining_horizon,
        )
        group_whittles[group][arm] = whittle_index

        # this list can contain dummy arms and order doesn't matter
        group_whittles_sorted[group].append(whittle_index)

        lambda_value = 1e6
        # Compute value function if this arm is active
        
        if active[original_arm_inds[arm]]:
          v_arm = self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )
          v_group += v_arm

          # add the dummy copies of this arm
          times_dummy = (group_dummy_arms[group]==original_arm_inds[arm]).sum()
          for _ in range(times_dummy):
            v_group += v_arm
            group_whittles_sorted[group].append(whittle_index)


      group_avg_b0[group] = v_group  # first round has no budget

      # if no active arms, assign 0 for b0 and b1 so MNW never considers
      if group_size_active == 0:
        group_avg_b0[group] = 0

      inds_sorted = np.argsort(group_whittles_sorted[group])[::-1]
      gws = np.array(group_whittles_sorted[group])
      group_whittles_sorted[group] = gws[inds_sorted]
      # inds_sorted = np.argsort(group_whittles[group])[::-1]
      # # gws = np.array(group_whittles_sorted[group])
      # group_whittles_sorted[group] = group_whittles[group][inds_sorted]

    # now that whittle indexes are computed, can compute initial deltas
    # init b1 and b0 to same, so undersized groups give -np.inf on log
    # Make sure to to check group sizes based on actual group size, not dummy
    group_avg_b1[:] = group_avg_b0[:]
    under_budget_groups = np.arange(len(self.unique_groups))[
        # group_budgets < group_sizes_active
        group_budgets < max_group_size  # allow all groups to reach max size
    ]
    for group in under_budget_groups:
      next_b = group_budgets[group] + 1
      if next_b < group_size_active:
        lambda_value = group_whittles_sorted[group][next_b-1: next_b+1].mean()
      else:
        lambda_value = group_whittles_sorted[group][next_b-1]

      group_inds = group_inds_dict[group]
      group_size = group_sizes[group]
      group_size_active = max_group_size  # group_sizes_active[group]
      t_probs = self.transition_probs[group_inds]
      r = self.rewards[group_inds]
      current_states = group_states[group]

      original_arm_inds = np.arange(self.n_arms)[group_inds]
      v_group = 0
      for arm in range(group_size):
        # Compute value function if this arm is active
        if active[original_arm_inds[arm]]:
          v_arm = self.convert_and_compute_v(
              current_states, lambda_value, t_probs, r, arm, remaining_horizon
          )
          v_group += v_arm

          # add the dummy copies of this arm
          times_dummy = (group_dummy_arms[group]==original_arm_inds[arm]).sum()
          for _ in range(times_dummy):
            v_group += v_arm

      vsum = sum(
          [(next_b) * (self.gamma**t) for t in range(remaining_horizon)]
      )
      group_avg_b1[group] = v_group + lambda_value * vsum

    # now compute log deltas
    # group_avg_log_deltas = np.log(group_avg_b1 - group_avg_b0)
    group_avg_log_deltas = (np.log(group_avg_b1) - np.log(group_avg_b0))

    # print('begin greedy selection up to', self.budget)
    # print(group_avg_b0)
    # print(group_avg_b1)
    # print(group_avg_log_deltas)
    full_budget_groups = group_budgets >= max_group_size  # group_sizes_active
    next_group_check = np.copy(group_avg_log_deltas)
    # so we don't allocate more budget to one group than arms in group
    next_group_check[full_budget_groups] = -np.inf
    for _ in range(self.budget):

      s1 = time.time()
      next_group_check = np.copy(group_avg_log_deltas)
      # so we don't allocate more budget to one group than arms in group
      next_group_check[full_budget_groups] = -np.inf
      # check for all inf here and break if so
      if (next_group_check == -np.inf).all():
        # print('all groups full')
        break
      mx_grp = np.argmax(next_group_check)
      group_budgets[mx_grp] += 1

      # only compute next value function if we have budget left
      if group_budgets[mx_grp] >= max_group_size:  # group_sizes_active[mx_grp]:
        group_avg_b0[mx_grp] = group_avg_b1[mx_grp]
      else:
        group_inds = group_inds_dict[mx_grp]
        group_size = group_sizes[mx_grp]
        group_size_active = max_group_size  # group_sizes_active[mx_grp]
        t_probs = self.transition_probs[group_inds]
        r = self.rewards[group_inds]
        current_states = group_states[mx_grp]

        original_arm_inds = np.arange(self.n_arms)[group_inds]

        # set lambda as the value between the last whittle index and the next
        next_b = group_budgets[mx_grp] + 1
        if next_b < group_size_active:
          lambda_value = group_whittles_sorted[mx_grp][next_b-1: next_b+1].mean()
        else:
          lambda_value = group_whittles_sorted[mx_grp][next_b-1]

        v_group = 0
        for arm in range(group_size):
          # Compute value function if this arm is active
          if active[original_arm_inds[arm]]:
            v_arm = self.convert_and_compute_v(
                current_states, lambda_value, t_probs, r, arm, remaining_horizon
            )
            v_group += v_arm

            # add the dummy copies of this arm
            times_dummy = (group_dummy_arms[mx_grp]==original_arm_inds[arm]).sum()
            for _ in range(times_dummy):
              v_group += v_arm

        group_avg_b0[mx_grp] = group_avg_b1[mx_grp]

        h = remaining_horizon
        vsum = sum([(next_b)*(self.gamma**t) for t in range(h)])
        group_avg_b1[mx_grp] = v_group + lambda_value*vsum

      gap = (np.log(group_avg_b1[mx_grp]) - np.log(group_avg_b0[mx_grp]))
      group_avg_log_deltas[mx_grp] = gap

      # print('budgets', group_budgets)
      # print('objectives', group_avg_log_deltas)
      # print(group_avg_b0)
      # print(group_avg_b1)
      # print(group_avg_b1 - group_avg_b0)
      # print(group_avg_log_deltas)
      full_budget_groups = group_budgets >= max_group_size  # group_sizes_active
      s2 = time.time()
      # print('One round water filling took %ss'%(s2-s1))

    # ranndomly allocate remaining budget
    if (next_group_check == -np.inf).all():
      unique_groups_np = np.array(self.unique_groups)
      # recompute group sizes without
      group_sizes = [
          (self.group_map == group).sum() for group in self.unique_groups
      ]
      b_used = group_budgets.sum()
      under_budget_groups = group_budgets < group_sizes
      for _ in range(b_used, self.budget):
        random_group = np.random.choice(unique_groups_np[under_budget_groups])
        group_budgets[random_group] += 1
        under_budget_groups = group_budgets < group_sizes

    # rescale budgets by group size
    # we effectively boosted them by max_group_size / group_size,
    # so now rescale budgets by the inverse: group_size / max_group_size
    # dealing with fractionals here is a choice... 
    # notes: groups can only decrease in size, not increase
    #    mulitply by the rescaling vector
    #    then normalize with the l2 norm
    #    the multiply by B
    #    then we deal with fractionals
    group_scale_factors =  group_sizes_active / max_group_size
    gbs_rescaled = np.copy(group_budgets)*group_scale_factors
    gbs_rescaled = gbs_rescaled / gbs_rescaled.sum() * self.budget
    # print('group budgets', group_budgets, group_budgets.sum())
    # print('group_scale_factors',group_scale_factors)
    # print('gbs_rescaled',gbs_rescaled)
    # check if any are over budget after rescaling
    gbs_running = np.copy(gbs_rescaled)
    over_budget_groups = gbs_running > group_sizes
    under_budget_groups = gbs_running < group_size
    while over_budget_groups.any():
      gbs_ceiled = np.copy(gbs_running)
      # print(full_budget_groups)
      # print(gbs_ceiled)
      # print(group_sizes)
      # ceil the over budget groups
      gbs_ceiled[over_budget_groups] = np.array(group_sizes)[over_budget_groups]
      # collect how much was reduced
      ceiled_out = (gbs_running - gbs_ceiled).sum()
      # how many under budget groups remain
      n_under_budget_groups = under_budget_groups.sum()
      # split the ceiled_out portion evenly among under budget groups
      budget_fraction_to_each = ceiled_out / n_under_budget_groups
      gbs_ceiled[under_budget_groups] += budget_fraction_to_each
      # make this our new running gb
      gbs_running = gbs_ceiled
      
      # If any group has now gone over budget, restart the process
      over_budget_groups = gbs_running > group_sizes
      under_budget_groups = gbs_running < group_size
    #    We will:
    #      1) store all decimals
    #      2) sum them
    #      3) randomly redistribute that sum across
    #         arms according to their remainders, without replacement
    #    If there are issues with small groups never getting picked
    #      we can create a floor at 1 potentially. but that'll likely bias
    #      to small groups again.
    gbs_running_int = gbs_running.astype(int)
    fractionals = gbs_running - gbs_running_int
    leftover_budget = self.budget - gbs_running_int.sum()
    if leftover_budget > 0:
      p = fractionals / fractionals.sum()
      plus_one_groups = np.random.choice(
          np.arange(len(self.unique_groups)),
          size=leftover_budget,
          p=p,
          replace=False
      )
      gbs_running_int[plus_one_groups] += 1

    # print('group_sizes', group_sizes)
    # print('group_sizes_active', group_sizes_active)
    # print('group budgets', group_budgets, group_budgets.sum())
    # print('group budgets rescaled', gbs_running_int, gbs_running_int.sum())

    # return group_budgets, group_whittles
    return gbs_running_int, group_whittles


# Prospective implementation. Use as baseline in multi-action cases.
class LagrangePolicy(RMABPolicy):
  """Implements the Lagrange Policy."""

  policy_key_str = 'Lagrange Policy'

  def __init__(self,
               n_arms: int,
               budget: int,
               transition_probs: np.ndarray,
               rewards: np.ndarray,
               lambda_lb: float = 0,
               lambda_ub: float = None,
               gamma: float = 0.9999,
               **kwargs,
               ) -> None:
    """Initialize the LagrangePolicy class.

    Computes the Lagrange policy for an RMAB. Multi-action.

    Args:
      n_arms: number of arms in the RMAB
      budget: action budget of the RMAB
      transition_probs: transition probabilities of the RMAB, of dimension
        n_arms x n_states x 2 x n_states
      rewards: rewards of the RMAB, of dimension n_arms x n_states
      lambda_lb: lower bound for solution value of lambda
      lambda_ub: upper bound for solution value of lambda
      gamma: discount factor rewards
      **kwargs: additional kwargs
    """
    super().__init__(n_arms, budget, transition_probs, rewards, **kwargs)
    self.lambda_lb = lambda_lb
    if lambda_ub is None:
      self.lambda_ub = rewards.max()/(1-gamma)
    else:
      self.labmda_ub = lambda_ub
    self.gamma = gamma
    self.n_states = transition_probs.shape[1]

    # Right now only 2-action environments exist in this repo,
    # so we don't yet pass in n_actions or cost_vector.
    # However, the rest of the class is implemented for multi-action.
    self.n_actions = 2
    self.cost_vector = [0, 1]

    self.lp_setup()

  @staticmethod
  def policy_key() -> str:
    """Return the name of the policy."""
    return LagrangePolicy.policy_key_str

  def lp_setup(self):
    """Set up a PuLP LP for later use."""
    dummy_state = np.zeros(self.n_states, dtype=np.int32)
    _, _, _, lp_resources = (
        mathprog_methods.lagrange_relaxation_infinite_horizon(
            self.transition_probs,
            self.rewards,
            self.cost_vector,
            self.budget,
            dummy_state,
            lambda_lim=self.lambda_ub,
            gamma=self.gamma,
        )
    )
    self.lp_resources = lp_resources

  def get_actions(self,
                  current_states: np.ndarray = None,
                  remaining_horizon: int = None) -> np.ndarray:
    """Returns actions according to the Equitable lp policy.

    Args:
      current_states: current state of all the arms
      remaining_horizon: number of time steps remanining in the simulation

    Returns:
      actions: n_arms-length array of actions to take according to the Whittle
        index policy
    """
    if current_states is None:
      raise ValueError(
          'current_states must be provided to LagrangePolicy.get_actions()')

    if remaining_horizon is None:
      raise ValueError(
          'remaining_horizon must be provided to LagrangePolicy.get_actions()'
      )

    print('Num states', self.n_states)
    q_values = np.zeros((self.n_arms, self.n_actions, self.n_states))
    print('Solving lp')
    v_values, lambda_star, _, _ = (
        mathprog_methods.lagrange_relaxation_infinite_horizon(
            self.transition_probs,
            self.rewards,
            self.cost_vector,
            self.budget,
            current_states,
            lambda_lim=self.lambda_ub,
            gamma=self.gamma,
            lp_resources=self.lp_resources,
        )
    )

    for n in range(self.n_arms):
      for a in range(self.n_actions):
        for s in range(self.n_states):
          q_values[n, a, s] = (
              self.rewards[n, s]
              - lambda_star * self.cost_vector[a]
              + self.gamma * v_values[n].dot(self.transition_probs[n, s, a])
          )

    qs_by_state = np.zeros((self.n_arms, self.n_actions))
    for n in range(self.n_arms):
      s = current_states[n]
      qs_by_state[n] = q_values[n, :, s]

    print('Solving Knapsack')
    decision_matrix = mathprog_methods.action_knapsack(
        qs_by_state, self.cost_vector, self.budget
    )
    actions = np.argmax(decision_matrix, axis=1)
    print('Done')

    return actions


class Config():
  """Config class for defining the list of policies that can be called."""
  # Define policy class mapping
  # If defining a new class, add its name to class_list
  class_list = [
      NoActionPolicy,
      EquitableLPPolicyMMR,
      EquitableLPPolicyMNW,
      EquitableLPPolicyMNWEqualGroups,
      HighA1cRRPolicy,
      RandomPolicy,
      RandomHighA1cPolicy,
      WhittleIndexPolicy,
  ]
  policy_class_map = {cl.policy_key(): cl for cl in class_list}

  @staticmethod
  def get_policy_class(policy_name):
    """Return a class reference corresponding the input policy string."""
    try:
      policy_class = Config.policy_class_map[policy_name]
    except KeyError as e:
      print(e)
      err_string = 'Did not recognize policy "%s". Please add it to '
      err_string += 'policies.Config.class_list'
      print(err_string % policy_name)
      exit()
    return policy_class

