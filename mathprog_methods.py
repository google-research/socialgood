"""Mathematical programming methods for implementing RMAB policies."""

import time

import numpy as np
import pulp


def var_to_value(v, shape, nested=True):
  """Convenience function to convert PuLP outputs to np arrays."""
  assert len(shape) <= 3

  v_vals = np.zeros(shape)
  if nested:
    if len(shape) == 1:
      for i in range(shape[0]):
        v_vals[i] = v[i].varValue
    if len(shape) == 2:
      for i in range(shape[0]):
        for j in range(shape[1]):
          v_vals[i][j] = v[i][j].varValue
    if len(shape) == 3:
      for i in range(shape[0]):
        for j in range(shape[1]):
          for k in range(shape[2]):
            v_vals[i][j][k] = v[i][j][k].varValue
  else:
    var_name_template = 'L_%i_%i_%i'
    for i in range(shape[0]):
      for j in range(shape[1]):
        for k in range(shape[2]):
          v_vals[i][j][k] = v[var_name_template%(i, j, k)].varValue

    return v_vals


def lagrange_relaxation_infinite_horizon(
    t, r, c, b, start_state, lambda_lim=None, gamma=0.95, lp_resources=None
):
  """LP for the Lagrange-relaxed value function, infinite horizon case."""
  # start = time.time()

  n_procs = t.shape[0]
  n_states = t.shape[1]
  n_actions = t.shape[2]

  if lp_resources is None:
    # Create a new model
    lp = pulp.LpProblem('LP_for_Hawkins_Lagrangian_relaxation', pulp.LpMinimize)

    # Create variables
    lb = 0
    ub = None
    if lambda_lim is not None:
      ub = lambda_lim

    index_var = pulp.LpVariable('index', lowBound=lb, upBound=ub)

    l_vars = pulp.LpVariable.dicts('l_vars', (range(n_procs), range(n_states)))

    # set constraints
    s = time.time()
    for p in range(n_procs):
      for i in range(n_states):
        for j in range(n_actions):
          lp += (
              l_vars[p][i] >= r[p][i] - index_var*c[j] + gamma*pulp.lpSum(
                  t[p, i, j, k] * l_vars[p][k] for k in range(n_states))
          )
    end = time.time()
    print('Constraint setup time', end - s)
  else:
    lp = lp_resources['lp']
    l_vars = lp_resources['l_vars']
    index_var = lp_resources['index_var']

  objective = pulp.lpSum(
      l_vars[i][start_state[i]] for i in range(n_procs)
  ) + index_var * (b) * ((1 - gamma) ** -1)

  # Objective
  lp.setObjective(objective)

  # Optimize model
  s = time.time()
  _ = lp.solve(pulp.PULP_CBC_CMD(msg=False))
  end = time.time()

  lambda_star = index_var.varValue
  l_vals = var_to_value(l_vars, (n_procs, n_states))

  lp_resources = {
      'lp': lp,
      'l_vars': l_vars,
      'index_var': index_var,
  }

  return l_vals, lambda_star, objective.value(), lp_resources


def lagrange_relaxation_finite_horizon(
    t, r, c, b, h, timestep, start_state, lambda_lim=None,
    gamma=0.95, lp_resources=None
):
  """LP for the Lagrange-relaxed value function, finite horizon case."""
  n_procs = t.shape[0]
  n_states = t.shape[1]
  n_actions = t.shape[2]

  if lp_resources is None:
    # Create a new model
    lp = pulp.LpProblem('LP_for_Hawkins_Lagrangian_relaxation', pulp.LpMinimize)

    # Create variables
    lb = 0
    ub = None
    if lambda_lim is not None:
      ub = lambda_lim

    index_var = pulp.LpVariable('index', lowBound=lb, upBound=ub)

    l_vars = pulp.LpVariable.dicts(
        'l_vars', (range(n_procs), range(n_states), range(h + 1))
    )

    # set constraints
    s = time.time()
    for p in range(n_procs):
      for i in range(n_states):
        for j in range(n_actions):
          for t in range(h):
            lp += (
                l_vars[p][i][t] >= r[p, i] - index_var*c[j] + gamma*pulp.lpSum(
                    t[p, i, j, k] * l_vars[p][k][t+1] for k in range(n_states)),
                'L_n%i_s%i_a%i_t%i'%(p, i, j, t)
            )
          lp += (
              l_vars[p][i][h] == r[p, i],
              'L_n%i_s%i_a%i_t%i'%(p, i, j, h)
              )
    end = time.time()
    print('Constraint setup time', end-s)
  else:
    lp = lp_resources['lp']
    l_vars = lp_resources['l_vars']
    index_var = lp_resources['index_var']

  # PuLP doesn't save variables in the same nested way you create them.
  # We can isolate the effects here though...
  # Check the format of the dict and load however its structured
  nested = True
  try:  # if original format
    objective = pulp.lpSum(
        l_vars[i][start_state[i]][timestep] for i in range(n_procs)
    ) + index_var * pulp.lpSum((b) * (gamma**t) for t in range(h - timestep))
  except Exception:  # if loaded from dictionary
    try:
      nested = False
      var_name_template = 'L_%i_%i_%i'
      objective = pulp.lpSum(
          l_vars[var_name_template % (i, start_state[i], timestep)]
          for i in range(n_procs)
      ) + index_var * pulp.lpSum(
          (b) * (gamma**t) for t in range(h - timestep)
      )
    except Exception as e2:
      print(e2)

  # Objective
  lp.setObjective(objective)

  # Optimize model
  s = time.time()
  _ = lp.solve(pulp.PULP_CBC_CMD(msg=False))
  end = time.time()
  print('solve time', end-s)
  print('objective value', objective.value())

  lambda_star = index_var.varValue
  l_vals = var_to_value(l_vars, (n_procs, n_states, h), nested=nested)

  lp_resources = {
      'lp': lp,
      'l_vars': l_vars,
      'index_var': index_var,
  }

  return l_vals, lambda_star, objective.value(), lp_resources


def action_knapsack(values, c, b):
  """Integer linear program to solve the action selection knapsack."""
  # values.shape == (N,A)
  assert c[0] == 0

  lp = pulp.LpProblem('Knapsack', pulp.LpMaximize)

  x = pulp.LpVariable.dicts(
      'x', (range(values.shape[0]), range(values.shape[1])), 0, 1, pulp.LpBinary
  )

  # Objective
  lp += (
      pulp.lpSum(
          pulp.lpSum(x[i][j] * c[j] for j in range(values.shape[1]))
          for i in range(values.shape[0])
      )
      == b
  )

  # set constraints
  lp += (
      pulp.lpSum(
          pulp.lpSum(x[i][j] * c[j] for j in range(values.shape[1]))
          for i in range(values.shape[0])
      )
      == b
  )

  for i in range(values.shape[0]):
    lp += pulp.lpSum(x[i][j] for j in range(values.shape[1])) == 1

  # Optimize model
  lp.solve(pulp.PULP_CBC_CMD(msg=False))

  x_out = var_to_value(x, values.shape)

  return x_out


