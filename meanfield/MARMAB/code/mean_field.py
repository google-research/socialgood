import pulp
import gurobipy as gb
from sklearn.cluster import KMeans
import numpy as np
import sys
import time


# DO NOT EDIT HERE. DO AT SOURCE
def mean_field_distribution(P, R, C, B, H, state, gamma,
                            sleeping_constraint, available_arms, sleeping_weeks):
  assert( (H-1, N_CLS, N_STATES, N_ACTIONS, N_STATES) == P.shape )
  assert( (N_CLS, N_STATES, N_ACTIONS) == R.shape )
  assert( (N_CLS, N_STATES, N_ACTIONS) == C.shape )
  assert( (H,) == B.shape )
  assert( (N_CLS, N_STATES) == state.shape )

  # the LP problem
  LP = pulp.LpProblem("Mean_Field", pulp.LpMaximize)

  mu = pulp.LpVariable.dicts("mu", (range(H),range(N_CLS),range(N_STATES)),
                            0, N_BEN, pulp.LpContinuous)
  alpha = pulp.LpVariable.dicts("alpha", (range(H),range(N_CLS),range(N_STATES),range(N_ACTIONS)),
                            0, N_BEN, pulp.LpContinuous)

  # Objective
  LP += (
      pulp.lpSum(alpha[t][i][s][a]*R[i,s,a]* (gamma**t)
                for t in range(H)
                for i in range(N_CLS)
                for s in range(N_STATES)
                for a in range(N_ACTIONS))
  )

  # Constraints
  # feasibility (equality)
  for t in range(H):
    for i in range(N_CLS):
      for s in range(N_STATES):
        LP += (
            pulp.lpSum(alpha[t][i][s][a] for a in range(N_ACTIONS)) == mu[t][i][s],
            f"feasibility for time {t}, arm {i}, state {s}"
        )

  # dynamics (equality)
  for t in range(H-1):
    for i in range(N_CLS):
      for sp in range(N_STATES):
        LP += (
            mu[t+1][i][sp] == pulp.lpSum(
                alpha[t][i][s][a]*P[t,i,s,a,sp]
                for s in range(N_STATES)
                for a in range(N_ACTIONS)
            ),
            f"transition dynamics for time {t}, arm {i}, state {sp}"
        )

  # budget (inequality)
  for t in range(H):
    LP += (
        pulp.lpSum(alpha[t][i][s][a]*C[i][s][a]
                  for i in range(N_CLS)
                  for s in range(N_STATES)
                  for a in range(N_ACTIONS)) <= B[t],
        f"budget constraint for time {t}"
    )

  # sleeping constraint
  if sleeping_constraint == True:
    # implemented only for one beneficiary per cluster & two actions
    assert N_CLS == N_BEN
    assert N_ACTIONS == 2
    assert sleeping_weeks > 0
    for i in range(N_CLS):
      for t in range(H):
        LP += (
            pulp.lpSum(alpha[t+sleep][i][s][1]
                        for sleep in range(np.minimum(sleeping_weeks+1,H-t))
                        for s in range(N_STATES)) <= 1,
            f"sleeping constraint for {i} in time duration [{t}...{t+sleeping_weeks}]"
        )

  # sleeping constraint 2
  if available_arms is not None:
    # implemented only for one beneficiary per cluster & two actions
    assert N_CLS == N_BEN
    assert N_ACTIONS == 2
    for i in range(N_CLS):
      for sleep in range(np.minimum(1-int(available_arms[i]), H)):
        for s in range(N_STATES):
          LP += (
              alpha[sleep][i][s][1] == 0
          )

  # initialization (equality)
  for i in range(N_CLS):
    for s in range(N_STATES):
      LP += (
          mu[0][i][s] == state[i][s],
          f"initialization for time 0, arm {i}, state {s}"
      )



  LP.solve(pulp.COIN_CMD(msg=False))

  # compute the numpy action
  alphaNow = np.zeros((N_CLS, N_STATES, N_ACTIONS))
  for i in range(N_CLS):
    for s in range(N_STATES):
      for a in range(N_ACTIONS):
        alphaNow[i][s][a] = alpha[0][i][s][a].varValue

  # due to some reason, likely floating point approximation in the LP solver,
  # action may have small negative values, which causes runtime errors later.
  # Set them to 0.
  alphaNow = np.maximum(alphaNow, 0)

  return alphaNow


# DO NOT EDIT HERE. DO AT SOURCE
def mean_field_distribution_gurobi(P, R, C, B, H, state, gamma,
                            sleeping_constraint, available_arms, sleeping_weeks):
  assert( (H-1, N_CLS, N_STATES, N_ACTIONS, N_STATES) == P.shape )
  assert( (N_CLS, N_STATES, N_ACTIONS) == R.shape )
  assert( (N_CLS, N_STATES, N_ACTIONS) == C.shape )
  assert( (H,) == B.shape )
  assert( (N_CLS, N_STATES) == state.shape )

  # the LP problem
  LP = gb.Model("Mean_Field")
  LP.modelSense = gb.GRB.MAXIMIZE

  mu = np.zeros((H, N_CLS, N_STATES), dtype=object)
  alpha = np.zeros((H, N_CLS, N_STATES, N_ACTIONS), dtype=object)

  for t in range(H):
    for i in range(N_CLS):
      for s in range(N_STATES):
        mu[t,i,s] = LP.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=N_BEN, name=f"mu_{t}_{i}_{s}")
        for a in range(N_ACTIONS):
          alpha[t,i,s,a] = LP.addVar(vtype=gb.GRB.CONTINUOUS, lb=0, ub=N_BEN,
                                     name=f"alpha_{t}_{i}_{s}_{a}")

  # Objective
  LP.setObjective(
      gb.quicksum(alpha[t][i][s][a]*R[i,s,a]* (gamma**t)
                for t in range(H)
                for i in range(N_CLS)
                for s in range(N_STATES)
                for a in range(N_ACTIONS))
    )

  # Constraints
  # feasibility (equality)
  for t in range(H):
    for i in range(N_CLS):
      for s in range(N_STATES):
        LP.addConstr(
            gb.quicksum(alpha[t][i][s][a] for a in range(N_ACTIONS)) == mu[t][i][s],
            f"feasibility for time {t}, arm {i}, state {s}"
        )

  # dynamics (equality)
  for t in range(H-1):
    for i in range(N_CLS):
      for sp in range(N_STATES):
        LP.addConstr(
            mu[t+1][i][sp] == gb.quicksum(
                alpha[t][i][s][a]*P[t,i,s,a,sp]
                for s in range(N_STATES)
                for a in range(N_ACTIONS)
            ),
            f"transition dynamics for time {t}, arm {i}, state {sp}"
        )

  # budget (inequality)
  for t in range(H):
    LP.addConstr(
        gb.quicksum(alpha[t][i][s][a]*C[i][s][a]
                  for i in range(N_CLS)
                  for s in range(N_STATES)
                  for a in range(N_ACTIONS)) <= B[t],
        f"budget constraint for time {t}"
    )

  # sleeping constraint
  if sleeping_constraint == True:
    # implemented only for one beneficiary per cluster & two actions
    assert N_CLS == N_BEN
    assert N_ACTIONS == 2
    assert sleeping_weeks > 0
    for i in range(N_CLS):
      for t in range(H):
        LP.addConstr(
            gb.quicksum(alpha[t+sleep][i][s][1]
                        for sleep in range(np.minimum(sleeping_weeks+1,H-t))
                        for s in range(N_STATES)) <= 1,
            f"sleeping constraint for {i} in time duration [{t}...{t+sleeping_weeks}]"
        )

  # sleeping constraint (borrowed from past time-steps)
  if available_arms is not None:
    # implemented only for one beneficiary per cluster & two actions
    assert N_CLS == N_BEN
    assert N_ACTIONS == 2
    for i in range(N_CLS):
      for sleep in range(np.minimum(1-int(available_arms[i]), H)):
        for s in range(N_STATES):
          LP.addConstr(
              alpha[sleep][i][s][1] == 0
          )

  # initialization (equality)
  for i in range(N_CLS):
    for s in range(N_STATES):
      LP.addConstr(
          mu[0][i][s] == state[i][s],
          f"initialization for time 0, arm {i}, state {s}"
      )


  LP.setParam('OutputFlag', False)
  LP.optimize()

  # compute the numpy action
  alphaNow = np.zeros((N_CLS, N_STATES, N_ACTIONS))
  for v in LP.getVars():
    if 'alpha' in v.varName:
      t = int(v.varName.split('_')[1])
      i = int(v.varName.split('_')[2])
      s = int(v.varName.split('_')[3])
      a = int(v.varName.split('_')[4])
      if t == 0:
        alphaNow[i][s][a] = v.x

  # due to some reason, likely floating point approximation in the LP solver,
  # action may have small negative values, which causes runtime errors later.
  # Set them to 0.
  alphaNow = np.maximum(alphaNow, 0)

  return alphaNow


# DO NOT EDIT HERE. DO AT SOURCE
def mean_field_action(alphaNow, state, C, sleeping_constraint, budget):
  # compute the discrete action (intervention) out of alpha
  # there are some quick hacks below due to fractional solution & rounding

  action = alphaNow.round().astype(int)

  # too many actions for a state
  for i in range(N_CLS):
    for s in range(N_STATES):
      action[i,s,0] += state[i,s] - action[i,s].sum()
      if action[i,s,0] < 0:
        # randomly select an action and decrease the count by 1
        # print("randomly select an action and decrease the count by 1")
        a = 1 + np.random.choice(np.arange(N_ACTIONS-1), p=action[i,s,1:]/action[i,s,1:].sum())
        action[i,s,a] -= 1
        action[i,s,0] += 1

  assert((action.sum(axis=-1) == state).all() and "1")

  # too many actions for budget
  too_many_actions = 0
  while (action*C).sum() > budget + 1e-3:
    p = action.copy()
    p[:,:,0] = 0  # setting 0 probability for cost 0 action (first one)
    p = p.reshape(p.size)
    ben = np.random.choice(np.arange(p.size), p = p/p.sum())
    i, s, a = np.unravel_index(ben, (N_CLS, N_STATES, N_ACTIONS))
    assert(a > 0)
    if action[i,s,a] > 0:
      action[i,s,a] -= 1
      action[i,s,0] += 1
    too_many_actions += 1
    # print(f"too_many_actions = {too_many_actions}")

  assert((action.sum(axis=-1) == state).all() and "2")

  # too few actions (only do if no sleeping constraint)
  if sleeping_constraint == False:
    # TODO: this can mess up with the sleeping constraint, avoid for now
    too_few_action = 0
    while (action*C).sum() < budget - 1e-3:
      p = action[:,:,0].copy()
      if p.sum() < 1:
        break  # too much budget
      p = p.reshape(p.size)
      ben = np.random.choice(np.arange(p.size), p=p/p.sum())
      i, s = np.unravel_index(ben, (N_CLS, N_STATES))
      a = np.random.randint(low=1, high=N_ACTIONS)
      action[i,s,a] += 1
      action[i,s,0] -= 1
      if (action*C).sum() > budget:
        action[i,s,a] -= 1
        action[i,s,0] += 1
        break
      too_few_action += 1
      # print(f"too_few_action = {too_few_action}")

  assert((action.sum(axis=-1) == state).all() and "3")

  return action


# DO NOT EDIT HERE. DO AT SOURCE
def mean_field_action_to_raw_action(cluster_labels, current_state_raw, action):
  if cluster_labels is None:
    # did not do any clustering, only one player per cluster
    # action for the player, action for everyone in the cluster
    action_raw = action.sum(axis=1)
    # one-hot to action index
    action_raw = action_raw.argmax(axis=-1)
  else:
    action_raw = np.zeros(N_BEN, dtype=int)
    for i in range(N_CLS):
      for s in range(N_STATES):
        # beneficiaries in this cluster and state
        bens = (np.arange(N_BEN))[(cluster_labels == i) & (current_state_raw == s)]
        # actions for this cluster and state
        acts = []; [ acts.extend([a]*action[i,s,a]) for a in range(N_ACTIONS) ]
        # randomly distribute the actions among the beneficiaries
        acts = np.random.permutation(acts)
        assert(acts.shape == bens.shape)
        action_raw[bens] = acts

  return action_raw


# DO NOT EDIT HERE. DO AT SOURCE
def mean_field_data_stat_to_non_stat(Pstat, H):
  P = np.zeros((H-1,) + Pstat.shape)
  P[:,:,:,:,:] = Pstat.reshape((1,) + Pstat.shape)

  B = np.zeros(H)
  B[:-1] = BUDGET

  return P, B




def mean_field_cluster(T_raw, R_raw, C_raw, B_raw, current_state_raw, n_clusters, cluster_labels):
  if n_clusters < 1 or n_clusters >= N_BEN:
    P = T_raw

    R = np.zeros((N_CLS, N_STATES, N_ACTIONS))
    R[:,:,:] = R_raw.reshape((N_CLS, N_STATES, 1))

    C = np.zeros((N_CLS, N_STATES, N_ACTIONS))
    C[:,:,:] = C_raw.reshape((1, 1, N_ACTIONS))

    Ni = np.ones(N_CLS)

    state = np.zeros((N_CLS, N_STATES))
    state[np.arange(N_CLS),current_state_raw] = 1

    cluster_labels = None

  else:
    if cluster_labels is None:
      kmeans = KMeans(n_clusters=N_CLS, random_state=0).fit(T_raw.reshape((N_BEN, N_STATES*N_ACTIONS*N_STATES)))
      cluster_labels = kmeans.labels_

    assert cluster_labels.max() < n_clusters

    P = np.zeros((N_CLS, N_STATES, N_ACTIONS, N_STATES))
    R = np.zeros((N_CLS, N_STATES, N_ACTIONS))
    C = np.zeros((N_CLS, N_STATES, N_ACTIONS))
    Ni = np.zeros(N_CLS, dtype=int)
    state = np.zeros((N_CLS, N_STATES), dtype=int)

    for i in range(N_CLS):
      bens_in_this_cls = (cluster_labels == i)
      Ni[i] = bens_in_this_cls.sum()
      if Ni[i] > 0:
        P[i,:,:,:] = T_raw[bens_in_this_cls, :,:,:].mean(axis=0)
        R[i,:,:] = (R_raw[bens_in_this_cls, :].mean(axis=0)).reshape((N_STATES,1))
      else:
        # set some default values, doesn't matter
        P[i,:,:,:] = 1/N_STATES
        R[i,:,:] = 0

    C[:,:,:] = C_raw.reshape((1, 1, N_ACTIONS))

    # some sanity steps (these may be required because of floating point approximations)
    P = np.maximum(0,P); P /= P.sum(axis=-1, keepdims=True)

    for i_ben in range(N_BEN):
      state[cluster_labels[i_ben], current_state_raw[i_ben]] += 1

  return cluster_labels, P, R, C, Ni, state



def mean_field_wrapper_jackson(T_raw, R_raw, C_raw, B_raw, current_state_raw, days_remaining_raw, gamma_raw=0.95, n_clusters=0, cluster_labels=None):
  """variable is for mean-field and variable_raw for the actual model"""
  # Modify Global constant (as they are not defined).
  # This function is used only in Aditya's code.
  global N_BEN, N_CLS, N_ACTIONS, N_STATES, BUDGET

  # Parameters
  (N_BEN, N_STATES, N_ACTIONS, _) = T_raw.shape
  BUDGET = B_raw
  if n_clusters < 1 or n_clusters >= N_BEN:
    N_CLS = N_BEN
  else:
    N_CLS = n_clusters

  assert( (N_BEN, N_STATES, N_ACTIONS, N_STATES) == T_raw.shape )
  assert( (N_BEN, N_STATES) == R_raw.shape )
  assert( (N_ACTIONS,) == C_raw.shape )
  assert( (N_BEN,) == current_state_raw.shape )

  H = days_remaining_raw
  gamma = gamma_raw

  # cluster (or not cluster) the data and convert to mean-field input
  cluster_labels, P, R, C, Ni, state = mean_field_cluster(T_raw, R_raw, C_raw, B_raw, current_state_raw, n_clusters, cluster_labels)

  P, B = mean_field_data_stat_to_non_stat(P, H)

  alphaNow = mean_field_distribution_gurobi(P=P, R=R, C=C, B=B, H=H, state=state,
                                       gamma=gamma, sleeping_constraint=False,
                                       available_arms=None, sleeping_weeks=None)
  action = mean_field_action(alphaNow=alphaNow, state=state, C=C,
                               sleeping_constraint=False, budget=B[0])

  # convert the mean-field output to simulator output
  action_raw = mean_field_action_to_raw_action(cluster_labels, current_state_raw, action)

  return action_raw, cluster_labels
