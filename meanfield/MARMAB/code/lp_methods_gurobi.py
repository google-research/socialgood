from gurobipy import *
import numpy as np 
import sys
import time




def sample_lam(T, R, C, B, start_states, gamma=0.95):

	timings = np.zeros(4)
	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for coupled Lagrangian relaxation")
	m.setParam( 'OutputFlag', False )

	V = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		mu[i, start_states[i]] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY

	index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')


	for p in range(NPROCS):
		for i in range(NSTATES):
			V[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='V_%s_%s'%(p,i))


	V = np.array(V)


	timings[0] = time.time() - start
	# print('Variables added in %ss:'%timings[0])
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state

	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	m.setObjectiveN(sum([V[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				m.addConstr( V[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], V[p]) )


	timings[1] = time.time() - start
	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()

	timings[2] = time.time() - start
	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	V_vals = np.zeros((NPROCS,NSTATES))

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

	return V_vals, index_solved_value, timings



def blam(T, R, C, B, lam_coeff_adjust, current_state, prev_model_data=None, lambda_lim=None, gamma=0.95):

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
		m = Model("LP for coupled Lagrangian relaxation, with grid search bounds")

		m.setParam( 'OutputFlag', False )

		V = np.zeros((NPROCS,NSTATES), dtype=object)

		# shape = (len(bound_grid-1), len(procs to bound))

		
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


		timings[0] = time.time() - start
		# print('Variables added in %ss:'%timings[0])
		start = time.time()


		m.modelSense=GRB.MINIMIZE

		# Set objective
		# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
		# minimze the value function

		# In Hawkins, only min the value function of the start state

		# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

		# min_feasible_coeff = min(C)/(1-gamma)
		lambda_coeff = B / (1-gamma)
		# print('lam_coeff_actual', lambda_coeff)


		# set constraints
		# print(NPROCS,NSTATES,NACTIONS)
		for p in range(NPROCS):
			for i in range(NSTATES):
				for j in range(NACTIONS):
					m.addConstr( V[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], V[p]) )



		# set variables and constraints for the bounded processes

		num_procs_bounded = V_slopes_in_ub.shape[0]
		num_grid_points = V_slopes_in_ub.shape[1]

		aux_vars_ub = np.zeros(num_procs_bounded, dtype=object)
		aux_var_constraints_ub = np.zeros((num_procs_bounded, num_grid_points), dtype=object)

		aux_vars_lb = np.zeros(num_procs_bounded, dtype=object)
		aux_var_constraints_lb = np.zeros((num_procs_bounded, num_grid_points), dtype=object)
		
		for p_ind,p in enumerate(sorted_inds):
			
			aux_vars_ub[p_ind] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='aux_var_ub_%s'%p)
			aux_vars_lb[p_ind] = m.addVar(vtype=GRB.CONTINUOUS, lb=-GRB.INFINITY, name='aux_var_lb_%s'%p)

			# TODO: Need to get all the z's into the objective function -- also why is it not solving?
			z_ub = aux_vars_ub[p_ind]
			z_lb = aux_vars_lb[p_ind]

			# print(V_slopes_in_lb[:num_grid_points_per_process[p],p],V_intercepts_in_lb[:num_grid_points_per_process[p],p])
			# print(V_slopes_in_ub[:num_grid_points_per_process[p],p],V_intercepts_in_ub[:num_grid_points_per_process[p],p])
			# print(bound_grid[:,p])
			# print(num_grid_points_per_process[p])
			# print()
			for i in range(num_grid_points_per_process[p]):
				aux_var_constraints_ub[p_ind,i] = m.addConstr( z_ub >= V_slopes_in_ub[p,i]*index_variable + V_intercepts_in_ub[p,i] )
				aux_var_constraints_lb[p_ind,i] = m.addConstr( z_lb >= V_slopes_in_lb[p,i]*index_variable + V_intercepts_in_lb[p,i] )


		# The -epsilon*index_variable term deals with the case when the value of lambda of is ambiguous
		# because it's not in the objective. We need to add a constraint to figure out how large lambda
		# can be made before the largest contributing Value function becomes 0
		# m.setObjectiveN(sum([V[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*lambda_coeff - index_variable*1e-10, 0, 1)
		m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*lambda_coeff + aux_vars_ub.sum())


		
		model_data = {
			'V':V,
			'lambda':index_variable,
			'm':m,

			'aux_vars_ub':aux_vars_ub,
			'aux_var_constraints_ub':aux_var_constraints_ub,

			'aux_vars_lb':aux_vars_lb,
			'aux_var_constraints_lb':aux_var_constraints_lb
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

		aux_vars_ub = prev_model_data['aux_vars_ub']
		aux_var_constraints_ub = prev_model_data['aux_var_constraints_ub']

		aux_vars_lb = prev_model_data['aux_vars_lb']
		aux_var_constraints_lb = prev_model_data['aux_var_constraints_lb']

		# load old vars
		V = np.zeros((NPROCS_TOTAL, prev_V.shape[1]),dtype=object)
		V[:num_procs_prev] = prev_V
		
		mu = np.zeros((NPROCS_TOTAL,prev_V.shape[1]),dtype=object)
		for i in range(NPROCS_TOTAL):
			mu[i, int(current_state[i])] = 1


		index_variable = prev_model_data['lambda']

		def remove(items):
			items = items.flatten(-1)
			filter_ind = [i for i,val in enumerate(items) if type(val)!=int ]
			items = items[filter_ind].tolist()
			m.remove(items)

		# remove unused aux_vars and constraints
		if NPROCS > 0:

			aux_vars_ub = aux_vars_ub[NPROCS:]
			aux_var_constraints_ub = aux_var_constraints_ub[NPROCS:]

			

			aux_vars_lb = aux_vars_lb[NPROCS:]
			aux_var_constraints_lb = aux_var_constraints_lb[NPROCS:]

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


		lambda_coeff = B / (1-gamma)
		# print('lambda_coeff_actual',lambda_coeff)

		# If we are seeking an upper bound
		if lb_or_ub == 'ub':
			# If there are any processes remaining to bound out
			if aux_vars_ub.shape[0] > 0:
				m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS_TOTAL)]) + index_variable*lambda_coeff + aux_vars_ub.sum())
			else:
				m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS_TOTAL)]) + index_variable*lambda_coeff)
		
		# If we are seeking a lower bound
		elif lb_or_ub == 'lb':
			if aux_vars_ub.shape[0] > 0:
				m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS_TOTAL)]) + index_variable*lambda_coeff + aux_vars_lb.sum())
			else:
				m.setObjective(sum([V[i].dot(mu[i]) for i in range(NPROCS_TOTAL)]) + index_variable*lambda_coeff)

		# add new constraints
		for p in range(num_procs_prev, NPROCS_TOTAL):
			for i in range(NSTATES):
				for j in range(T.shape[2]):
					p_adj = p - num_procs_prev
					m.addConstr( V[p][i] >= R[p_adj][i] - index_variable*c[j] + gamma*LinExpr(T[p_adj,i,j], V[p]) )


		# don't need to mess with the constraints of aux vars, that's handled in objective



		model_data = {
			'V':V,
			'lambda':index_variable,
			'm':m,

			'aux_vars_ub':aux_vars_ub,
			'aux_var_constraints_ub':aux_var_constraints_ub,

			'aux_vars_lb':aux_vars_lb,
			'aux_var_constraints_lb':aux_var_constraints_lb,
			'sorted_inds':sorted_inds
			
		}


	timings[1] = time.time() - start

	start = time.time()

	# Optimize model

	m.optimize()

	timings[2] = time.time() - start
	start = time.time()


	V_vals = np.zeros((NPROCS_TOTAL, V.shape[1]))

	index_solved_value = 0
	for v in m.getVars():
		# print('%s %g' % (v.varName, v.x))
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'V' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			V_vals[i,j] = v.x



	timings[3] = time.time() - start

	start = time.time()

	model_data['obj'] = m.getObjective().getValue()

	return V_vals, index_solved_value, timings, model_data







# https://dspace.mit.edu/handle/1721.1/29599
def hawkins(T, R, C, B, start_state, lambda_lim=None, gamma=0.95):

	start = time.time()

	NPROCS = T.shape[0]
	NSTATES = T.shape[1]
	NACTIONS = T.shape[2]

	# Create a new model
	m = Model("LP for Hawkins Lagrangian relaxation")
	m.setParam( 'OutputFlag', False )

	L = np.zeros((NPROCS,NSTATES),dtype=object)
	
	mu = np.zeros((NPROCS,NSTATES),dtype=object)
	for i in range(NPROCS):
		# mu[i] = np.random.dirichlet(np.ones(NSTATES))
		mu[i, int(start_state[i])] = 1

	c = C

	# Create variables
	lb = 0
	ub = GRB.INFINITY
	if lambda_lim is not None:
		ub = lambda_lim

	index_variable = m.addVar(vtype=GRB.CONTINUOUS, lb=lb, ub=ub, name='index')


	for p in range(NPROCS):
		for i in range(NSTATES):
			L[p,i] = m.addVar(vtype=GRB.CONTINUOUS, name='L_%s_%s'%(p,i))


	L = np.array(L)


	# print('Variables added in %ss:'%(time.time() - start))
	start = time.time()


	m.modelSense=GRB.MINIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first
	# minimze the value function

	# In Hawkins, only min the value function of the start state
	# print(current_state)
	# m.setObjectiveN(sum([L[i][current_state[i]] for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	m.setObjectiveN(sum([L[i].dot(mu[i]) for i in range(NPROCS)]) + index_variable*B*((1-gamma)**-1), 0, 1)

	# set constraints
	for p in range(NPROCS):
		for i in range(NSTATES):
			for j in range(NACTIONS):
				# m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*L[p].dot(T[p,i,j]) )
				m.addConstr( L[p][i] >= R[p][i] - index_variable*c[j] + gamma*LinExpr(T[p,i,j], L[p])) 



	# print('Constraints added in %ss:'%(time.time() - start))
	start = time.time()

	# Optimize model

	m.optimize()
	# m.printStats()

	# print('Model optimized in %ss:'%(time.time() - start))
	start = time.time()


	L_vals = np.zeros((NPROCS,NSTATES))

	index_solved_value = 0
	for v in m.getVars():
		if 'index' in v.varName:
			index_solved_value = v.x

		if 'L' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			L_vals[i,j] = v.x

	# print('Variables extracted in %ss:'%(time.time() - start))
	start = time.time()

	return L_vals, index_solved_value






# Transition matrix, reward vector, action cost vector
def action_knapsack(values, C, B):


	m = Model("Knapsack")
	m.setParam( 'OutputFlag', False )

	c = C

	x = np.zeros(values.shape, dtype=object)

	for i in range(x.shape[0]):
		for j in range(x.shape[1]):
			x[i,j] = m.addVar(vtype=GRB.BINARY, name='x_%i_%i'%(i,j))



	m.modelSense=GRB.MAXIMIZE

	# Set objective
	# m.setObjectiveN(obj, index, priority) -- larger priority = optimize first

	# minimze the value function
	m.setObjectiveN((x*values).sum(), 0, 1)

	# set constraints
	m.addConstr( x.dot(C).sum() <= B )
	for i in range(values.shape[0]):
		# m.addConstr( x[i].sum() <= 1 )
		m.addConstr( x[i].sum() == 1 )


	# Optimize model

	m.optimize()

	x_out = np.zeros(x.shape)

	for v in m.getVars():
		if 'x' in v.varName:
			i = int(v.varName.split('_')[1])
			j = int(v.varName.split('_')[2])

			x_out[i,j] = v.x

		else:
			pass
			# print((v.varName, v.x))


	return x_out
