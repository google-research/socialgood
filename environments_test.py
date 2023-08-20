"""Tests for environments.py."""

import environments
import numpy as np


class DiabetesAppStreamingEnvironmentTests:
  """Tests for DiabetesAppStreamingEnvironment."""

  def __init__(self, n_arms, start_seed, horizon, alpha, stream_map,
               n_simulations, env, model_data_file):
    self.n_arms = n_arms
    self.start_seed = start_seed
    self.horizon = horizon
    self.alpha = alpha
    self.stream_map = stream_map
    self.n_simulations = n_simulations
    self.model_data_file = model_data_file

    self.env = env

    self.eng_health_rewards = self.env.group_rewards[0]
    self.arm_probs = self.env.arm_probs

    # (s_e, s_m, s_t, s_d, s_h, s_c)
    ret_dat = env.get_state_dicts_combined()
    self.state_tup_to_ind = ret_dat[0]
    self.state_ind_to_tup = ret_dat[1]

  def print_outcome(self, start, states_expected, states_observed):
    print('Start state:', start)
    print()

    print('states expected:', states_expected)
    print('states observed:', states_observed)
    print()
    print('states expected')
    self.print_state_defs(states_expected)
    print('states observed')
    self.print_state_defs(states_observed)

  def print_state_defs(self, state_inds):
    for s in state_inds:
      print('Index: %s,\tTuple: %s' % (s, self.state_ind_to_tup[s]))

  def test_state_transitions_from_start_no_action(self):
    """Test state transition from start state, no action."""
    env = self.env

    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    # * must go to maintenance
    # * may or may not observe
    # * if observe, then health state may change
    all_next_states = []

    # Maintance, same health state
    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # Maintance, change health state
    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # No observation
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 1
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed

  def test_state_transitions_from_start_all_act(self):
    """Test state transition from start state, act all arms."""
    env = self.env

    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    actions[:] = 1
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    # * can go to engaged or maintenance
    # * may or may not observe
    # * if observe, then health state may change
    all_next_states = []

    # stay engaged, same health
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # stay engaged, change health
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # maintenance, observe, same health
    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # maintenance, observe, change health
    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # maintenance, no observe
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = 1
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed

  def test_state_transitions_st1_maint_all_act(self):
    """Test state transition from t1, act on all arms."""
    env = self.env

    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 1
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    actions[:] = 1
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    all_next_states = []

    ## observe cases
    # re-engage, change health
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # re-engage, same health
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # maintenance, change health
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # maintenance, same health
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.ENGAGED)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    ## no observation
    # maintenance
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 2
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    # dropout
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 2
    s_d = env.DROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed

  def test_state_transitions_dropout(self):
    """Test state transition from dropout state."""
    env = self.env

    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 4
    s_d = env.DROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    start = (s_e, s_m, s_t, s_d, s_h, s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    actions[:2] = 1
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    all_next_states = []

    # Can only stay in dropout
    s_e = env.MAINTENANCE
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 5
    s_d = env.DROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed

  def run_reward_test(self, state_tup, env, reward_expected, arm):
    """Helper function to run reward tests."""

    epsilon = 1e-4

    state_ind = self.state_tup_to_ind[state_tup]
    actions = np.zeros(self.n_arms, dtype=np.int32)

    env.current_states[:] = state_ind
    _, rewards_observed = env.step(actions)
    reward_observed = rewards_observed[arm]

    str_data = (state_tup, reward_observed, reward_expected)
    print('state: %s, R_obs: %s, R_expected: %s' % str_data)

    assert abs(reward_observed - reward_expected) < epsilon

  def test_engagement_state_rewards(self):
    """Test only engagement rewards."""
    alpha = 1.0  # all engagement reward
    env = environments.DiabetesAppPartialObsEnvironment(
        n_arms=self.n_arms,
        start_seed=self.start_seed,
        horizon=self.horizon,
        alpha=alpha,
        stream_map=self.stream_map,
        model_data_file=self.model_data_file,
    )

    test_arm_ind = 0

    # Engaged reward
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_eng']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Maintenance reward (belief chain logic)
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 1
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_maintenance']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Dropout reward (s_d = DROPOUT)
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 2
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_dropout']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Maintenance reward
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_maintenance']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Dropout reward at different timepoint, with different chain head
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 1
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_dropout']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Dropout reward at different timepoint, with different chain head
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 2
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_dropout']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Maintenance reward, different timepoint, not reading from chain values.
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 4
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_maintenance']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

  def evolve_belief(self, health_belief, env, eng_sequence, arm_n):
    """Helper function to evolve belief state."""
    for eng_state in eng_sequence:
      if eng_state == env.ENGAGED:
        p_gl = self.arm_probs[arm_n]['p_eng_gl']
        p_ll = self.arm_probs[arm_n]['p_eng_ll']
      elif eng_state != env.ENGAGED:
        p_gl = self.arm_probs[arm_n]['p_noeng_gl']
        p_ll = self.arm_probs[arm_n]['p_noeng_ll']
      health_belief = env.evolve_belief(health_belief, p_gl, p_ll)
    return health_belief

  def test_health_state_rewards(self):
    """Test only health rewards."""
    alpha = 0  # all engagement reward
    env = environments.DiabetesAppPartialObsEnvironment(
        n_arms=self.n_arms,
        start_seed=self.start_seed,
        horizon=self.horizon,
        alpha=alpha,
        stream_map=self.stream_map,
        model_data_file=self.model_data_file,
    )

    test_arm_ind = 0

    # A1c > 8
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # A1c < 8
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_a1clt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # A1c > 8, different chain head
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # A1c < 8, different chain head
    s_e = env.MAINTENANCE
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    reward_expected = self.eng_health_rewards['r_a1clt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c < 8, evolve belief
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 1
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [env.MAINTENANCE]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    reward_expected = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    reward_expected += (1-belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c < 8, evolve belief
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 2
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [env.MAINTENANCE, env.MAINTENANCE]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    reward_expected = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    reward_expected += (1-belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c > 8, evolve belief
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 3
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [env.MAINTENANCE, env.MAINTENANCE, env.ENGAGED]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    reward_expected = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    reward_expected += (1-belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c < 8, evolve belief
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = 4
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [
        env.MAINTENANCE,
        env.MAINTENANCE,
        env.ENGAGED,
        env.MAINTENANCE,
    ]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    reward_expected = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    reward_expected += (1-belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c > 8 and dropout, evolve belief
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 5
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [
        env.MAINTENANCE,
        env.ENGAGED,
        env.ENGAGED,
        env.MAINTENANCE,
        env.MAINTENANCE,
    ]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    reward_expected = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    reward_expected += (1-belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

  def test_combined_rewards(self):
    r"""Test combined health and engagement reward when alpha \in (0, 1)."""
    alpha = 0.42  # some weight on both engagement and health
    env = environments.DiabetesAppPartialObsEnvironment(
        n_arms=self.n_arms,
        start_seed=self.start_seed,
        horizon=self.horizon,
        alpha=alpha,
        stream_map=self.stream_map,
        model_data_file=self.model_data_file,
    )

    test_arm_ind = 0

    # Start from A1c > 8 and dropout, evolve belief
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 5
    s_d = env.DROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [
        env.MAINTENANCE,
        env.ENGAGED,
        env.ENGAGED,
        env.MAINTENANCE,
        env.MAINTENANCE,
    ]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    h_reward_exp = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    h_reward_exp += (1 - belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    e_reward_exp = self.eng_health_rewards['r_dropout']
    reward_expected = e_reward_exp * alpha + h_reward_exp * (1 - alpha)
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c < 8, evolve belief
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 2
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = [env.MAINTENANCE, env.ENGAGED]
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    h_reward_exp = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    h_reward_exp += (1 - belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    e_reward_exp = self.eng_health_rewards['r_maintenance']
    reward_expected = e_reward_exp * alpha + h_reward_exp * (1 - alpha)
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c > 8
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 0
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = []
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    h_reward_exp = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    h_reward_exp += (1 - belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    e_reward_exp = self.eng_health_rewards['r_eng']
    reward_expected = e_reward_exp * alpha + h_reward_exp * (1 - alpha)
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

    # Start from A1c < 8
    s_e = env.ENGAGED
    s_m = (env.ENGAGED, env.MAINTENANCE)
    s_t = 0
    s_d = env.NODROPOUT
    s_h = env.A1CLT8
    s_c = env.ARM_ACTIVE
    state_tup = (s_e, s_m, s_t, s_d, s_h, s_c)
    health_belief = s_h
    eng_sequence = []
    belief_a1clt8 = self.evolve_belief(
        health_belief, env, eng_sequence, test_arm_ind
    )
    h_reward_exp = belief_a1clt8 * self.eng_health_rewards['r_a1clt8']
    h_reward_exp += (1 - belief_a1clt8) * self.eng_health_rewards['r_a1cgt8']
    e_reward_exp = self.eng_health_rewards['r_eng']
    reward_expected = e_reward_exp * alpha + h_reward_exp * (1 - alpha)
    self.run_reward_test(state_tup, env, reward_expected, test_arm_ind)

  def test_streaming_state_transitions_sc1(self):
    """Test transitions from inactive streaming state to active state."""
    env = self.env

    s_c = 1
    start = env.get_inactive_state(s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    all_next_states = []

    # Go to this state when first become active
    s_e = env.ENGAGED
    s_m = (env.MAINTENANCE, env.MAINTENANCE)
    s_t = env.OBSERVE_RESET
    s_d = env.NODROPOUT
    s_h = env.A1CGT8
    s_c = env.ARM_ACTIVE
    next_state = (s_e, s_m, s_t, s_d, s_h, s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed

  def test_streaming_state_transitions_sc2(self):
    """Test transitions from inactive streaming stats to another."""
    env = self.env

    s_c = 2
    start = env.get_inactive_state(s_c)
    state_ind = self.state_tup_to_ind[start]
    actions = np.zeros(self.n_arms, dtype=np.int32)
    results = np.zeros((self.n_arms, self.n_simulations), dtype=np.int32)

    for i in range(self.n_simulations):
      env.current_states[:] = state_ind
      next_state_inds, _ = env.step(actions)
      results[:, i] = next_state_inds

    # all possible next states
    all_next_states = []

    # Reduce stream counter by 1
    s_c = 1
    next_state = env.get_inactive_state(s_c)
    next_state_tup = self.state_tup_to_ind[next_state]
    all_next_states.append(next_state_tup)

    states_expected = set(all_next_states)
    states_observed = set(results.reshape(-1))

    self.print_outcome(start, states_expected, states_observed)
    assert states_expected == states_observed


if __name__ == '__main__':

  print('Running environments.py tests')

  n_arms_test = 5
  start_seed_test = 0
  horizon_test = 12
  alpha_test = 0.5
  n_simulations_test = 100000
  model_data_file_test = 'test_data/db_app_test.csv'
  stream_map_test = np.array([0, 0, 1, 2, 3])

  env_test = environments.DiabetesAppPartialObsEnvironment(
      n_arms_test,
      start_seed_test,
      horizon_test,
      model_data_file_test,
      stream_map_test,
      alpha=alpha_test,
  )

  tests = DiabetesAppStreamingEnvironmentTests(
      n_arms_test,
      start_seed_test,
      horizon_test,
      alpha_test,
      stream_map_test,
      n_simulations_test,
      env_test,
      model_data_file_test,
  )

  print('Test 1: From start, no action')
  tests.test_state_transitions_from_start_no_action()
  print()

  print('Test 1.1: From start, all act')
  tests.test_state_transitions_from_start_all_act()
  print()

  print('Test 2: From s_t = 1, maintenance, all act ')
  tests.test_state_transitions_st1_maint_all_act()
  print()

  print('Test 3: Start from Dropout')
  tests.test_state_transitions_dropout()
  print()

  print('Test 4: Engagement Rewards')
  tests.test_engagement_state_rewards()
  print()

  print('Test 5: Health Rewards')
  tests.test_health_state_rewards()
  print()

  print('Test 6: Combined Rewards')
  tests.test_combined_rewards()
  print()

  print('Test 7+8: Exiting streaming')
  tests.test_streaming_state_transitions_sc1()
  tests.test_streaming_state_transitions_sc2()
  print()

  print('All tests passed.')
