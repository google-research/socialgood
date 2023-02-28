import pulp
from sklearn.cluster import KMeans
import numpy as np 
import sys
import time


def var_to_value(V, shape):
	assert(len(shape) <= 2)

	V_vals = np.zeros(shape)
	if len(shape) == 1:
		for i in range(shape[0]):
			V_vals[i] = V[i].varValue
	if len(shape) == 2:
		for i in range(shape[0]):
			for j in range(shape[1]):
				V_vals[i][j] = V[i][j].varValue

	return V_vals


def sample_lam(T, R, C, B, start_state, gamma=0.95):

	timings = np.zeros(4)
	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	LP = pulp.LpProblem("LP_for_coupled_Lagrangian_relaxation", pulp.LpMinimize)

	# Create variables
	lb = 0
	ub = None

	index_variable = pulp.LpVariable('index', lowBound=lb, upBound=ub)

	V = pulp.LpVariable.dicts("V", (range(NPROCS), range(NSTATES)))

	timings[0] = time.time() - start
	# print('Variables added in %ss:'%timings[0])
	start = time.time()

	# Objective
	LP += (
		pulp.lpSum(V[i][start_state[i]] for i in range(NPROCS))
		+ index_variable*B*((1-gamma)**-1)
	)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				LP += (
					V[p][i] >= R[p][i] - index_variable*C[j] + gamma*pulp.lpSum(
						T[p,i,j,k]*V[p][k] for k in range(NSTATES))
				)

	timings[1] = time.time() - start
	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model
	LP.solve(pulp.PULP_CBC_CMD(msg=False))

	timings[2] = time.time() - start
	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()

	index_solved_value = index_variable.varValue
	V_vals = var_to_value(V, (NPROCS, NSTATES))

	timings[3] = time.time() - start
	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return V_vals, index_solved_value, timings



def blam(T, R, C, B, lam_coeff_adjust, current_state, prev_model_data=None, lambda_lim=None, gamma=0.95):

	# Not converted to PuLP
	assert(False)

	timings = np.zeros(4)
	start = time.time()

	model_data = {}

	V = None
	NPROCS_TOTAL = 0

	# Create a new model
	if prev_model_data == None:

		NPROCS = T.shape[0]
		NPROCS_TOTAL = NPROCS

		NSTATES = T.shape[1]
		NACTIONS = T.shape[2]
		m = Model("LP for coupled Lagrangian relaxation, with bounds ")

		m.setParam( 'OutputFlag', False )

		V = np.zeros((NPROCS,NSTATES),dtype=object)
		
		mu = np.zeros((NPROCS,NSTATES),dtype=object)
		for i in range(NPROCS):
			mu[i, int(current_state[i])] = 1  

		c = C

		# Create variables
		lb = 0
		ub = GRB.INFINITY
		if lambda_lim is not None:
			ub = lambda_lim

		index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')


		for p in range(NPROCS):
			for i in range(NSTATES):
				V[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='V_%s_%s'%(p,i))


		V = np.array(V)


		timings[0] = time.time() - start
		# print('Variables added in %ss:'%timings[0])
		start = time.time()


		m.modelSense=GRB.MINIMIZE

		
		lambda_coeff = lam_coeff_adjust + B / (1-gamma)

		# The -epsilon*index_variable term deals with the case when the value of lambda of is ambiguous
		# because it's not in the objective. We need to add a constraint to figure out how large lambda
		# can be made before the largest contributing Value function becomes 0
		# m.setObjectiveN(sum([V[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*lambda_coeff - index_variable*1e-10, 0, 1)
		m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*lambda_coeff)

		# set constraints
		# print(NPROCS,NSTATES,NACTIONS)
		for p in range(NPROCS):
			for i in range(NSTATES):
				for j in range(NACTIONS):
					m.addConstr( V[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], V[p]) )

		
		model_data = {
			'V':V,
			'lambda':index_variable,
			'm':m
		}

	else:

		m = prev_model_data['m']
		m.reset(0)

		m.setParam( 'OutputFlag', False )
		NPROCS = len(T)

		prev_V = prev_model_data['V']
		num_procs_prev = prev_V.shape[0]

		NSTATES = prev_V.shape[1]

		NPROCS_TOTAL = num_procs_prev + NPROCS

		# load old vars
		V = np.zeros((NPROCS_TOTAL, prev_V.shape[1]),dtype=object)
		V[:num_procs_prev] = prev_V
		
		mu = np.zeros((NPROCS_TOTAL,prev_V.shape[1]),dtype=object)
		for i in range(NPROCS_TOTAL):
			# mu[i] = np.random.dirichlet(np.ones(prev_V.shape[1]))
			mu[i, int(current_state[i])] = 1

		index_variable = prev_model_data['lambda']

		c = C

		# Create variables
		lb = 0
		ub = GRB.INFINITY

		

		# create new vars
		for p in range(num_procs_prev, NPROCS_TOTAL):
			for i in range(NSTATES):
				V[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='V_%s_%s'%(p,i))


		timings[0] = time.time() - start
		# print('Variables added in %ss:'%timings[0])
		start = time.time()


		m.modelSense=GRB.MINIMIZE

		# Set objective
		# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
		# minimze the value function

		# In Hawkins, only min the value function of the start state

		# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

		min_feasible_coeff = min(C)/(1-gamma)
		lambda_coeff = max(lam_coeff_adjust + B / (1-gamma), min_feasible_coeff)
		# print('lambda_coeff_actual',lambda_coeff)
		
		# m.setObjectiveN(sum([V[i].dot(mu[i]) for i in range(num_procs_prev+NPROCS)]) + index_variable*lambda_coeff - index_variable*1e-10, 0, 1)
		# m.setObjective(sum([V[i].dot(mu[i]) for i in range(num_procs_prev+NPROCS)]) + index_variable*lambda_coeff - index_variable*1e-10)
		m.setObjective(sum([V[i].dot(mu[i]) for i in range(num_procs_prev+NPROCS)]) + index_variable*lambda_coeff)

		# add new constraints
		for p in range(num_procs_prev, NPROCS_TOTAL):
			for i in range(NSTATES):
				for j in range(T.shape[2]):
					p_adj = p - num_procs_prev
					m.addConstr( V[p][i] >= R[p_adj][i] - index_variable*c[j] + gamma*LinExpr(T[p_adj,i,j], V[p]) )

		model_data = {
			'V':V,
			'lambda':index_variable,
			'm':m,
		}


	timings[1] = time.time() - start
	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# if not prev_model_data == None:
	# 	1/0

	timings[2] = time.time() - start
	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	V_vals = np.zeros((NPROCS_TOTAL, V.shape[1]))

	index_solved_value = 0
	for v in m.getVars():
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'V' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			V_vals[i,j] = v.x

	timings[3] = time.time() - start
	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	model_data['obj'] = m.getObjective().getValue()

	return V_vals, index_solved_value, timings, model_data




# BLam with gridsearch
def blam_gridsearch(T, R, V_slopes_in_ub, V_intercepts_in_ub, V_slopes_in_lb, V_intercepts_in_lb,
					 num_grid_points_per_process, C, B, current_state, sorted_inds, lb_or_ub, 
					 prev_model_data=None, lambda_lim=None, gamma=0.95):

	M = 1e6

	# print("num procs")
	# print(T.shape)
	# print("num bounded out")
	# print(V_slopes_in.shape)


	timings = np.zeros(4)
	start = time.time()

	model_data = {}

	V = None
	NPROCS_TOTAL = 0

	# Create a new model
	# Must pass in nonempty T and R if creating new model
	if prev_model_data == None:

		NPROCS = T.shape[0]
		NPROCS_TOTAL = NPROCS

		NSTATES = T.shape[1]
		NACTIONS = T.shape[2]

		LP = pulp.LpProblem("LP_for_coupled_Lagrangian_relaxation,_with_grid_search_bounds", pulp.LpMinimize)

		# Create variables
		lb = 0
		ub = None
		if lambda_lim is not None:
			ub = lambda_lim

		# Variables
		index_variable = pulp.LpVariable('index', lowBound=lb, upBound=ub)
		V = pulp.LpVariable.dicts("V", (range(NPROCS), range(NSTATES)))

		# set variables for the bounded processes
		num_procs_bounded = V_slopes_in_ub.shape[0]
		num_grid_points = V_slopes_in_ub.shape[1]

		aux_vars_ub = pulp.LpVariable.dicts("aux_vars_ub", range(num_procs_bounded))
		aux_vars_lb = pulp.LpVariable.dicts("aux_vars_lb", range(num_procs_bounded))
		
		# timing
		timings[0] = time.time() - start
		start = time.time()

		# Objective
		lambda_coeff = B / (1-gamma)
		# The -epsilon*index_variable term deals with the case when the value of lambda of is ambiguous
		# because it's not in the objective. We need to add a constraint to figure out how large lambda
		# can be made before the largest contributing Value function becomes 0
		LP += (
			pulp.lpSum(V[i][current_state[i]] for i in range(NPROCS))
			+ index_variable*lambda_coeff 
			+ pulp.lpSum(aux_vars_ub[i] for i in range(num_procs_bounded))
		)

		# set constraints
		for p in range(NPROCS):
			for i in range(NSTATES):
				for j in range(NACTIONS):
					LP += (
						V[p][i] >= R[p][i] - index_variable*C[j] + gamma*pulp.lpSum(
							T[p,i,j,k]*V[p][k] for k in range(NSTATES))
					)


		# set constraints for the bounded processes
		# trying to store the constraints... will this work for pulp?
		# removing them and adding again in the next step
		# aux_var_constraints_ub = np.zeros((num_procs_bounded, num_grid_points), dtype=object)
		# aux_var_constraints_lb = np.zeros((num_procs_bounded, num_grid_points), dtype=object)
		
		for p_ind,p in enumerate(sorted_inds):
			for i in range(num_grid_points_per_process[p]):
				LP += (
					aux_vars_ub[p_ind] >= V_slopes_in_ub[p,i]*index_variable + V_intercepts_in_ub[p,i]
				)
				LP += (
					aux_vars_lb[p_ind] >= V_slopes_in_lb[p,i]*index_variable + V_intercepts_in_lb[p,i]
				)
		
		model_data = {
			'V': var_to_value(V, (NPROCS, NSTATES)),
			'lambda': index_variable.varValue,
			'aux_vars_ub': var_to_value(aux_vars_ub, (num_procs_bounded,)),
			'aux_vars_lb': var_to_value(aux_vars_lb, (num_procs_bounded,)),

			'sorted_inds': sorted_inds,
			'V_slopes_in_ub': V_slopes_in_ub,
            'V_intercepts_in_ub': V_intercepts_in_ub,
            'V_slopes_in_lb': V_slopes_in_lb,
            'V_intercepts_in_lb': V_intercepts_in_lb,
			'num_grid_points_per_process': num_grid_points_per_process
		}

	else:

		LP = pulp.LpProblem("LP_for_coupled_Lagrangian_relaxation,_with_grid_search_bounds", pulp.LpMinimize)

		NPROCS = len(T)

		prev_V = prev_model_data['V']
		num_procs_prev = prev_V.shape[0]

		NSTATES = prev_V.shape[1]

		NPROCS_TOTAL = num_procs_prev + NPROCS

		# load old vars & create new
		V = pulp.LpVariable.dicts("V", (range(NPROCS_TOTAL), range(NSTATES)))
		for i in range(num_procs_prev):
			for j in range(NSTATES):
				V[i][j].setInitialValue(prev_V[i][j])
		
		index_variable = pulp.LpVariable('index')
		index_variable.setInitialValue(prev_model_data['lambda'])

		# def remove(items):
		# 	items = items.flatten(-1)
		# 	filter_ind = [i for i,val in enumerate(items) if type(val)!=int ]
		# 	items = items[filter_ind].tolist()
		# 	m.remove(items)

		aux_vars_len = 0
		# If there are any processes remaining to bound out
		if NPROCS < prev_model_data['aux_vars_ub'].shape[0]:
			aux_vars_len = prev_model_data['aux_vars_ub'].shape[0] - NPROCS
			aux_vars_ub = pulp.LpVariable.dicts("aux_vars_ub", range(aux_vars_len))
			aux_vars_lb = pulp.LpVariable.dicts("aux_vars_lb", range(aux_vars_len))

			for i in range(aux_vars_len):
				aux_vars_ub[i].setInitialValue(prev_model_data['aux_vars_ub'][i + NPROCS])
				aux_vars_lb[i].setInitialValue(prev_model_data['aux_vars_lb'][i + NPROCS])


		timings[0] = time.time() - start
		# print('Variables added in %ss:'%timings[0])
		start = time.time()


		lambda_coeff = B / (1-gamma)
		# print('lambda_coeff_actual',lambda_coeff)

		# If we are seeking an upper bound
		if lb_or_ub == 'ub':
			# If there are any processes remaining to bound out
			if aux_vars_len:
				LP += (
					pulp.lpSum(V[i][current_state[i]] for i in range(NPROCS_TOTAL))
					+ index_variable*lambda_coeff 
					+ pulp.lpSum(aux_vars_ub[i] for i in range(aux_vars_len))
				)
			else:
				LP += (
					pulp.lpSum(V[i][current_state[i]] for i in range(NPROCS_TOTAL))
					+ index_variable*lambda_coeff 
				)
		
		# If we are seeking a lower bound
		elif lb_or_ub == 'lb':
			if aux_vars_len:
				LP += (
					pulp.lpSum(V[i][current_state[i]] for i in range(NPROCS_TOTAL))
					+ index_variable*lambda_coeff 
					+ pulp.lpSum(aux_vars_lb[i] for i in range(aux_vars_len))
				)
			else:
				LP += (
					pulp.lpSum(V[i][current_state[i]] for i in range(NPROCS_TOTAL))
					+ index_variable*lambda_coeff 
				)

		# add new constraints
		for p in range(num_procs_prev, NPROCS_TOTAL):
			for i in range(NSTATES):
				for j in range(T.shape[2]):
					p_adj = p - num_procs_prev
					LP += ( 
						V[p][i] >= R[p_adj][i] - index_variable*C[j] + gamma*pulp.lpSum(
							T[p_adj,i,j,k]*V[p][k] for k in range(NSTATES)) 
					)



		for p_ind, p in enumerate(sorted_inds):
			p_aux = p_ind + aux_vars_len - len(sorted_inds)
			if p_aux >= 0: # we have already taken care of other aux_vars
				for i in range(prev_model_data['num_grid_points_per_process'][p]):
					LP += (
						aux_vars_ub[p_aux] 
						>= prev_model_data['V_slopes_in_ub'][p,i]*index_variable 
						+ prev_model_data['V_intercepts_in_ub'][p,i]
					)
					LP += (
						aux_vars_lb[p_aux] 
						>= prev_model_data['V_slopes_in_lb'][p,i]*index_variable 
						+ prev_model_data['V_intercepts_in_lb'][p,i]
					)



		model_data = prev_model_data
		model_data['V'] = var_to_value(V, (NPROCS_TOTAL, NSTATES))
		model_data['lambda'] = index_variable.varValue
		model_data['aux_vars_ub'] = var_to_value(aux_vars_ub, (aux_vars_len,))
		model_data['aux_vars_lb'] = var_to_value(aux_vars_lb, (aux_vars_len,))


	timings[1] = time.time() - start

	start = time.time()

	# Optimize model
	LP.solve(pulp.PULP_CBC_CMD(msg=False, warmStart=True))

	timings[2] = time.time() - start
	start = time.time()


	V_vals = var_to_value(V, (NPROCS_TOTAL, NSTATES))

	index_solved_value = index_variable.varValue

	timings[3] = time.time() - start

	start = time.time()

	model_data['obj'] = LP.objective.value()

	return V_vals, index_solved_value, timings, model_data







# https://dspace.mit.edu/handle/1721.1/29599
def hawkins(T, R, C, B, start_state, lambda_lim=None, gamma=0.95):
	# start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	LP = pulp.LpProblem("LP_for_Hawkins_Lagrangian_relaxation", pulp.LpMinimize)

	# Create variables
	lb = 0
	ub = None
	if lambda_lim is not None:
		ub = lambda_lim

	index_variable = pulp.LpVariable('index', lowBound=lb, upBound=ub)

	L = pulp.LpVariable.dicts("L", (range(NPROCS), range(NSTATES)))

	# Objective
	LP += (
		pulp.lpSum(L[i][start_state[i]] for i in range(NPROCS))
		+ index_variable*B*((1-gamma)**-1)
	)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				LP += (
					L[p][i] >= R[p][i] - index_variable*C[j] + gamma*pulp.lpSum(
						T[p,i,j,k]*L[p][k] for k in range(NSTATES))
				)

	# Optimize model
	status = LP.solve(pulp.PULP_CBC_CMD(msg=False))
	# print(pulp.LpStatus[status])

	index_solved_value = index_variable.varValue
	L_vals = var_to_value(L, (NPROCS, NSTATES))

	return L_vals, index_solved_value






def action_knapsack(values, C, B):

	# values.shape == (N,A)
	assert(C[0] == 0)

	LP = pulp.LpProblem("Knapsack", pulp.LpMaximize)

	x = pulp.LpVariable.dicts("x", (range(values.shape[0]), range(values.shape[1])), 
							0, 1, pulp.LpContinuous)

	# Objective
	LP += (
		pulp.lpSum(x[i][j]*values[i][j] 
			for i in range(values.shape[0])
			for j in range(values.shape[1]))
	)

	# set constraints
	LP += ( pulp.lpSum (
		pulp.lpSum( x[i][j]*C[j] for j in range(values.shape[1]) ) 
		for i in range(values.shape[0])) <= B )

	for i in range(values.shape[0]):
		LP += ( pulp.lpSum( x[i][j] for j in range(values.shape[1]) ) == 1 )

	# Optimize model
	LP.solve(pulp.PULP_CBC_CMD(msg=False))

	x_out = var_to_value(x, values.shape)

	# we have solved an LP but want an integer solution.
	# claim: at max one variable will be fractional
	x_int = np.maximum(0, np.round(x_out)).astype(int)
	assert ((np.abs(x_out - x_int) > 1e-3).sum() <= 2)
	# randomly remove a costly action
	while (x_int * C.reshape((1,C.size))).sum() > B:
		p = (values*x_int)[:,1:] / (C[1:].reshape(1, C.size-1) + 1e-3)
		p = p.reshape(p.size)
		idx = np.random.choice(np.arange(p.size), p=p/p.sum())
		i, a = np.unravel_index(idx, (x_int.shape[0], x_int.shape[1]-1))
		a += 1
		x_int[i,a] = 0; x_int[i,0] = 1

	# print(f"x_out {x_out} and x_int {x_int}")

	return x_int




def mean_field_wrapper_jackson(*args, **kwargs):
	"""See the implementation in mean_field.py"""
	import mean_field
	return mean_field.mean_field_wrapper_jackson(*args, **kwargs)