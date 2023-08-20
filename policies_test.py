"""Tests for policies.py."""

from environments import DiabetesAppPartialObsEnvironment
import numpy as np
from policies import RandomHighA1cPolicy
from policies import WhittleIndexPolicy


class RandomHighA1cPolicyTests:
  """Tests for the RandomHighA1cPolicy policy."""

  def __init__(
      self,
      n_arms,
      start_seed,
      horizon,
      alpha,
      stream_map,
      n_simulations,
      env,
      policy,
  ):
    self.n_arms = n_arms
    self.start_seed = start_seed
    self.horizon = horizon
    self.alpha = alpha
    self.stream_map = stream_map
    self.n_simulations = n_simulations

    self.env = env
    self.policy = policy

    # (s_e, s_m, s_t, s_d, s_h, s_c)
    ret_data = env.get_state_dicts_combined()
    self.state_tup_to_ind = ret_data[0]
    self.state_ind_to_tup = ret_data[1]

  def test_act_on_higha1c_only_and_randomly(self):
    """Test if policy is acting randomly on high A1c arms."""
    env = self.env
    eps = 1e-2

    actions = np.zeros((self.n_simulations, self.n_arms), dtype=np.int32)
    states_inds = np.zeros(self.n_arms, dtype=np.int32)
    states_s_h = np.zeros(self.n_arms, dtype=np.int32)
    states_of_arms_acted = np.zeros(
        (self.n_simulations, self.policy.budget), dtype=np.int32
    )

    for i in range(self.n_simulations):
      for n in range(self.n_arms):
        # randomly sample states
        s_e = np.random.choice([env.ENGAGED, env.MAINTENANCE])
        s_m = tuple(
            np.random.choice(
                [env.ENGAGED, env.MAINTENANCE], size=2, replace=True
            )
        )
        s_t = np.random.choice(np.arange(env.horizon))
        s_d = np.random.choice([env.NODROPOUT, env.DROPOUT])

        # Ensure that there are budget + 1 higha1c arms
        # And make them the same every time, so that we are only testing
        # randomness of the policy
        if n < self.policy.budget+1:
          s_h = env.A1CGT8
        else:
          s_h = env.A1CLT8
        s_c = env.ARM_ACTIVE
        states_s_h[n] = s_h

        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        state_ind = self.state_tup_to_ind[state_tup]
        states_inds[n] = state_ind

      # get actions
      action_vector = self.policy.get_actions(current_states=states_inds)
      actions[i] = action_vector

      # get health states of arms that were acted on
      states_of_arms_acted[i] = states_s_h[action_vector.astype(bool)]

    # all arms acted on should have high a1c
    high_a1c_inds = states_of_arms_acted == self.policy.high_a1c_code
    num_high_a1c_actions = high_a1c_inds.sum()
    expected_num_high_a1c_actions = self.n_simulations * self.policy.budget
    high_a1c_check = num_high_a1c_actions == expected_num_high_a1c_actions
    print('Acted only on High A1c arms?')
    print('%i/%i'%(num_high_a1c_actions, expected_num_high_a1c_actions))

    action_distro = actions.sum(axis=0) / actions.sum()
    print('action distro', action_distro)
    b_adj = self.policy.budget+1
    str_data = (b_adj, 1/b_adj, self.n_arms - b_adj)
    print('should be %s entries of %.2f followed by %i zeros'%str_data)

    # action distro for the first b+1 arms should be uniformly 1/(b+1)
    firstb = action_distro[:self.policy.budget+1]
    randomness_check = (abs(firstb - 1 / (self.policy.budget + 1)) < eps).all()
    # action distro for the last n-b arms should be uniformly 0
    remaining = action_distro[self.policy.budget+1:]
    randomness_check &= remaining.mean() < eps

    assert randomness_check
    assert high_a1c_check

  def test_act_on_higha1c_then_randomly_on_lowa1c(self):
    """Test if policy acts first on higha1c, then randomly on lowa1c."""
    env = self.env
    eps = 1e-2

    actions = np.zeros((self.n_simulations, self.n_arms), dtype=np.int32)
    states_inds = np.zeros(self.n_arms, dtype=np.int32)
    states_s_h = np.zeros(self.n_arms, dtype=np.int32)
    states_of_arms_acted = np.zeros(
        (self.n_simulations, self.policy.budget), dtype=np.int32
    )

    for i in range(self.n_simulations):
      for n in range(self.n_arms):
        # randomly sample states
        s_e = np.random.choice([env.ENGAGED, env.MAINTENANCE])
        s_m = tuple(
            np.random.choice(
                [env.ENGAGED, env.MAINTENANCE], size=2, replace=True
            )
        )
        s_t = np.random.choice(np.arange(env.horizon))
        s_d = np.random.choice([env.NODROPOUT, env.DROPOUT])
        if n == 0:
          s_h = env.A1CGT8
        else:
          s_h = env.A1CLT8
        s_c = env.ARM_ACTIVE
        states_s_h[n] = s_h

        state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
        state_ind = self.state_tup_to_ind[state_tup]
        states_inds[n] = state_ind

      # get actions
      action_vector = self.policy.get_actions(current_states=states_inds)
      actions[i] = action_vector

      # get health states of arms that were acted on
      states_of_arms_acted[i] = states_s_h[action_vector.astype(bool)]

    # Check that first arm was always acted on
    first_arm_always_acted = actions[:, 0].sum() == self.n_simulations

    # remaining distribution should be even split of the remaining budget
    action_distro = actions.sum(axis=0) / actions.sum() * self.policy.budget
    print('action distro', action_distro)
    budget_adj = self.policy.budget - 1
    arms_adj = self.n_arms - 1
    print('should be a single 1 followed by all %.2f'%(budget_adj/arms_adj))
    remaining = action_distro[1:]
    randomness_check = (abs(remaining - budget_adj/arms_adj) < eps).all()

    assert randomness_check
    assert first_arm_always_acted


class WhittleIndexPolicyTests:
  """Tests for the WhittleIndexPolicy policy."""

  def __init__(
      self,
      n_arms,
      start_seed,
      horizon,
      alpha,
      stream_map,
      n_simulations,
      env,
      policy,
  ):
    self.n_arms = n_arms
    self.start_seed = start_seed
    self.horizon = horizon
    self.alpha = alpha
    self.stream_map = stream_map
    self.n_simulations = n_simulations

    self.env = env
    self.policy = policy

    # (s_e, s_m, s_t, s_d, s_h, s_c)
    ret_data = env.get_state_dicts_combined()
    self.state_tup_to_ind = ret_data[0]
    self.state_ind_to_tup = ret_data[1]

  def test_whittle_se_maintenance(self):
    """Test Whittlex for start state Maintenance."""
    env = self.env

    states = np.zeros(self.n_arms, dtype=np.int32)

    # create start state
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    states[:] = state_ind

    whittle_indexes = self.policy.compute_whittle_indexes_for_states(
        current_states=states, remaining_horizon=self.horizon
    )

    assert whittle_indexes[0] > 0

  def test_whittle_se_engaged(self):
    """Test Whittlex for start state Engaged."""
    env = self.env

    states = np.zeros(self.n_arms, dtype=np.int32)

    # create start state
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    states[:] = state_ind

    whittle_indexes = self.policy.compute_whittle_indexes_for_states(
        current_states=states, remaining_horizon=self.horizon
    )

    assert whittle_indexes[0] > 0

  def test_whittle_se_dropout(self):
    """Test Whittlex for dropout state."""
    env = self.env

    states = np.zeros(self.n_arms, dtype=np.int32)

    # create start state
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 1
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    states[:] = state_ind

    whittle_indexes = self.policy.compute_whittle_indexes_for_states(
        current_states=states, remaining_horizon=self.horizon
    )

    assert whittle_indexes[0] <= 0

  def test_whittle_inactive(self):
    """Test Whittlex for inactive start state."""
    env = self.env

    states = np.zeros(self.n_arms, dtype=np.int32)

    # create start state
    start = env.get_inactive_state(1)
    state_ind = self.state_tup_to_ind[start]
    states[:] = state_ind

    whittle_indexes = self.policy.compute_whittle_indexes_for_states(
        current_states=states, remaining_horizon=self.horizon
    )

    assert whittle_indexes[0] <= 0


if __name__ == '__main__':
  print('Running policies.py tests')

  n_arms_test = 5
  budget_test = 2
  start_seed_test = 0
  horizon_test = 3
  alpha_test = 1
  n_simulations_test = 20000
  model_data_file_test = 'test_data/db_app_test.csv'
  # all arms start on round 0
  stream_map_test = np.zeros(n_arms_test, dtype=np.int32)

  env_test = DiabetesAppPartialObsEnvironment(
      n_arms=n_arms_test,
      start_seed=start_seed_test,
      horizon=horizon_test,
      alpha=alpha_test,
      stream_map=stream_map_test,
      model_data_file=model_data_file_test,
  )

  environment_rewards = env_test.get_reward_definition()
  environment_probs = env_test.transition_probs

  policy_test = RandomHighA1cPolicy(
      n_arms=n_arms_test,
      budget=budget_test,
      state_index_to_tuple=env_test.state_index_to_tuple,
      high_a1c_code=env_test.A1CGT8,
      state_tuple_health_index=env_test.HEALTH_INDEX,
  )

  tests = RandomHighA1cPolicyTests(
      n_arms_test,
      start_seed_test,
      horizon_test,
      alpha_test,
      stream_map_test,
      n_simulations_test,
      env_test,
      policy_test,
  )

  policy_test = WhittleIndexPolicy(
      n_arms=n_arms_test,
      budget=budget_test,
      transition_probs=environment_probs,
      rewards=environment_rewards,
      index_lb=-1.0,
      index_ub=1.0,
      gamma=1.0,
      binary_search_tolerance=1e-1,
  )

  whittle_tests = WhittleIndexPolicyTests(
      n_arms_test,
      start_seed_test,
      horizon_test,
      alpha_test,
      stream_map_test,
      n_simulations_test,
      env_test,
      policy_test,
  )

  print('Test 1: Acting randomly and only on High A1c, if enough available?')
  tests.test_act_on_higha1c_only_and_randomly()
  print()

  print('Test 2: Acting first on high A1c arms, then randomly on remaining?')
  tests.test_act_on_higha1c_then_randomly_on_lowa1c()
  print()

  print('Test 3: Whittle index tests')
  whittle_tests.test_whittle_se_maintenance()
  whittle_tests.test_whittle_se_engaged()
  whittle_tests.test_whittle_inactive()
  whittle_tests.test_whittle_se_dropout()
  print()

  print('All tests passed.')
