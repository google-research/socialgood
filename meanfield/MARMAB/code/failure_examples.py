import numpy as np
import matplotlib.pyplot as plt

import simulation_environments


def get_rmab(leakS = 0.05, leakR = 0.1, leakD = 0.2):
	# print(leakS, leakR, leakD)
	global N_STATES, N_ACTIONS, STATE_NAMES
	STATE_NAMES = ["Re-S", "Re-E", "Gr-S", "Gr-E", "Drop"]
	# get the parameters
	P, R, C, B, start_state = simulation_environments.get_adverse_for_lagrangian(4, leakS, leakR, leakD)


	# we only need one MDP, every arm is supposed to be homogenous
	P = P.mean(axis=0)
	R = R.mean(axis=0)

	(N_STATES, N_ACTIONS, _) = P.shape
	assert N_ACTIONS == 2, "We are supposed to work with RMABs with 2 actions."
	assert (N_STATES, N_ACTIONS, N_STATES) == P.shape
	assert (N_STATES,) == R.shape

	# don't need C, B, and start_state for computing the index and reward

	R_old = R.copy()
	R = np.zeros((N_STATES, N_ACTIONS))
	R[:,:] = R_old.reshape((N_STATES,1))

	# print(P,R)
	return P, R


def index(task, P, R):
	"""
	We have two cases:
	if task = "get":  # Get Indices
		then find the index for each of the states
		return an array of size N_STATES with the indices
	else if task = "test":  # Test Indexability
		loop through a range of lambda values and get the 
			difference b/w active vs passive actions
		return array of size <range_of_lambda> x N_STATES
	"""
	assert task in ["test", "get"]

	if task == "test":
		LD = np.arange(-5, 5, 0.01)
		V_diff = np.zeros((LD.size, N_STATES))
		for i in range(LD.size):
			ld = LD[i]

			R_ld = R.copy()
			R_ld[:,0] += ld
			pi, V, Q, k = policy_iteration(P, R_ld)
			# print(f"Ran task={task}, ld={ld}, in iterations={k}")
			V_diff[i,:] = Q[:,1] - Q[:,0]

		return LD, V_diff
	elif task == "get":
		TOL = 1e-5
		indices = np.zeros(N_STATES)
		
		for s in range(N_STATES):
			ld_min = -100; ld_max = 100
			while True:
				ld = (ld_min + ld_max)/2
				R_ld = R.copy()
				R_ld[:,0] += ld
				pi, V, Q, k = policy_iteration(P, R_ld)
				# print(f"Ran task={task}, state={s}, ld={ld}, in iterations={k}")
				if Q[s,1] - Q[s,0] > TOL:
					ld_min = ld
				elif  Q[s,1] - Q[s,0] < -TOL:
					ld_max = ld
				else:
					break
			indices[s] = ld

		return indices


def policy_iteration(P, R):
	"""
	Computes the optimal policy for the given MDP.

	Input:
	P: transition probabilities for each SxA
	R: reward for each SxA pair

	Output:
	pi0 (S -> A): policy, i.e., the recommended action for each state
	V (S -> real): v-value for each state
	Q (SxA -> real): q-value for each SxA pair
	k: time till convergence of this function (policy iteration algorithm)
	"""
	assert (N_STATES, N_ACTIONS, N_STATES) == P.shape
	assert (N_STATES, N_ACTIONS) == R.shape

	# pi0 = np.array(N_STATES, dtype=int)
	pi = np.zeros(N_STATES, dtype=int)
	V = np.zeros(N_STATES)

	for k in range(1000):
		pi0 = pi.copy()
		V0 = V.copy()

		# policy evaluation
		A = np.zeros((N_STATES+1, N_STATES+1))  # the last variable is for rho
		b = np.zeros(N_STATES+1)

		for s in range(N_STATES):
			A[s, N_STATES] = 1  # for rho
			A[s, s] = 1  # for V(s)
			A[s, :N_STATES] -= P[s, pi0[s], :]  # for V(y) for every y
			b[s] = R[s, pi0[s]]

		# normalize the last state to have V(drop) = 0
		A[N_STATES, N_STATES-1] = 1
		b[N_STATES] = 0

		# sol = np.linalg.lstsq(A,b,rcond=None)
		# V = sol[0][:N_STATES]
		# rho = sol[0][N_STATES]
		sol = np.linalg.solve(A,b)
		V = sol[:N_STATES]
		rho = sol[N_STATES]

		# policy improvement
		Q = np.zeros((N_STATES, N_ACTIONS))
		for s in range(N_STATES):
			Q[s,:] = R[s,:] + np.matmul(P[s,:,:], V)
			pi[s] = np.argmax(Q[s,:]).astype(int)

		if (pi0 == pi).all() or (np.absolute(V - V0) < 1e-5).all():
			break

	# print(k)

	return pi0, V, Q, k


def simulate_state_evolution(policy, P, R, horizon):
	"""
	Takes the policy and the MDP and simulates the differential
	equation that corresponds to the limiting case of number of
	arms tend to infinity.

	Input:
	policy: action for each state
	P: transition probabilities for each SxA
	R: reward for each SxA pair

	Output:
	T: the sequence of the time points
	Z: the sequence of states evaluated at the time points
	"""
	assert policy.size == N_STATES

	BUDGET = 0.5
	dt = 1
	T = horizon
	z0 = np.array([0.5, 0, 0.5, 0, 0])

	H = int(T/dt)
	Z = np.zeros((H+1, N_STATES))
	Z[0,:] = z0
	for h in range(H):
		Z[h+1,:] = diff_eq_next_step(Z[h,:], policy, dt, P, BUDGET)

	return np.arange(0, T+dt, dt), Z


def diff_eq_next_step(z, policy, dt, P, remaining_budget):
	"""
	A helper function for 'simulate_state_evolution' that computes the next state. 
	"""
	assert N_ACTIONS == 2
	assert remaining_budget <= 1

	u = np.zeros((N_STATES, N_ACTIONS))

	for s in policy:
		u[s, 1] = min(remaining_budget, z[s])
		u[s, 0] = z[s] - u[s, 1]
		remaining_budget -= u[s, 1]

	assert remaining_budget == 0
	assert (np.abs(u.sum(axis=-1) - z) < 1e-5).all()

	z_new = (1 - dt) * z + dt * (
		np.matmul(u[:,0].reshape((1,N_STATES)), P[:,0,:])
		+ np.matmul(u[:,1].reshape((1,N_STATES)), P[:,1,:])).reshape((N_STATES,))

	# print(f"u = {u}, z_new = {z_new}")
	# input("hi")

	return z_new


def run_limit_cycle_experiment():
	P, R = get_rmab(0.05, 0.1, 0.2)  # note that reward for actions 0 and 1 is the same

	gamma = 0.95
	horizon = 1000

	# Arguably better policy for finite time / discounted reward
	policy = np.array([1, 0, 2, 4, 3], dtype=int)
	T, Z = simulate_state_evolution(policy, P, R, horizon)
	reward = (R[:,0].reshape(1,N_STATES) * Z * (gamma**T).reshape(T.size,1)).sum()
	print(f"Alternate. policy = {policy}, reward = {reward}")
	print(f"Z (stable) = {Z[-1,:]}")

	for s in range(N_STATES):
		plt.plot(T, Z[:,s], label=STATE_NAMES[s])
	plt.legend(loc="best")
	plt.show()

	# Whittle policy
	indices = index("get", P, R)
	policy = np.flip(np.argsort(indices))
	T, Z = simulate_state_evolution(policy, P, R, horizon)
	reward = (R[:,0].reshape(1,N_STATES) * Z * (gamma**T).reshape(T.size,1)).sum()
	print(f"Whittle. policy = {policy}, reward = {reward}")
	print(f"Z (stable) = {Z[-1,:]}")

	for s in range(N_STATES):
		plt.plot(T, Z[:,s], label=STATE_NAMES[s])
	plt.legend(loc="best")
	plt.show()


def run_indexability_experiment():
	P, R = get_rmab(0.05, 0.1, 0.2) 
	
	print(index("get", P, R))

	LD, V_diff = index("test", P, R)

	plt.plot(LD, np.zeros(LD.shape), '-', label='0')
	for s in range(N_STATES):
		plt.plot(LD, V_diff[:,s], label=STATE_NAMES[s])
	plt.legend(loc="best")
	plt.show()



def get_reward(gamma, horizon, leakS, leakR, leakD):
	P, R = get_rmab(leakS, leakR, leakD)  # note that reward for actions 0 and 1 is the same

	alt_policy = np.array([1, 0, 2, 4, 3], dtype=int)

	# Arguably better policy for finite time / discounted reward
	T, Z = simulate_state_evolution(alt_policy, P, R, horizon)
	alt_reward = (R[:,0].reshape(1,N_STATES) * Z * (gamma**T).reshape(T.size,1)).sum()

	# Whittle policy
	indices = index("get", P, R)
	whittle_policy = np.flip(np.argsort(indices))
	T, Z = simulate_state_evolution(whittle_policy, P, R, horizon)
	whittle_reward = (R[:,0].reshape(1,N_STATES) * Z * (gamma**T).reshape(T.size,1)).sum()

	return alt_reward, whittle_reward



def run_reward_vs_horizon():
	gamma = 1
	leak = 0.1

	m = 200
	Horizons = np.arange(m) + 5
	Alt = np.zeros(m)
	Whittle = np.zeros(m)

	for i in range(m):
		horizon = Horizons[i]
		Alt[i], Whittle[i] = get_reward(gamma, horizon, leak, leak, leak)

	plt.plot(Horizons, Alt/Horizons, label='Alternate')
	plt.plot(Horizons, Whittle/Horizons, label='Whittle')
	plt.legend()
	plt.xlabel("Time Horizon")
	plt.ylabel("Average Reward")
	plt.show()

	plt.plot(Horizons, Whittle/Alt)
	plt.xlabel("Time Horizon")
	plt.ylabel("Ratio of Average Reward (Whittle/Alternate)")
	plt.show()


def run_reward_vs_gamma():
	horizon = 1000
	leak = 0.1

	m = 100
	Gammas = (np.arange(m) + 4*m) / (5*m)
	Alt = np.zeros(m)
	Whittle = np.zeros(m)

	for i in range(m):
		gamma = Gammas[i]
		Alt[i], Whittle[i] = get_reward(gamma, horizon, leak, leak, leak)

	plt.plot(Gammas, Alt, label='Alternate')
	plt.plot(Gammas, Whittle, label='Whittle')
	plt.legend()
	plt.xlabel("Discount Factor")
	plt.ylabel("Discounted Reward")
	plt.show()

	plt.plot(Gammas, Whittle/Alt)
	plt.xlabel("Discount Factor")
	plt.ylabel("Ratio of Discounted Reward (Whittle/Alternate)")
	plt.show()


def run_reward_vs_mixing():
	gamma = 0.95
	horizon = 1000

	m = 100
	Leaks = (np.arange(m) + 1) / (m + 2)
	Alt = np.zeros(m)
	Whittle = np.zeros(m)

	for i in range(m):
		leak = Leaks[i]
		Alt[i], Whittle[i] = get_reward(gamma, horizon, leak, leak, leak)

	plt.plot(Leaks, Alt, label='Alternate')
	plt.plot(Leaks, Whittle, label='Whittle')
	plt.legend()
	plt.xlabel("eta (decreases as mixing time increases)")
	plt.ylabel("Discounted Reward")
	plt.show()

	plt.plot(Leaks, Whittle/Alt)
	plt.xlabel("eta (decreases as mixing time increases)")
	plt.ylabel("Ratio of Discounted Reward (Whittle/Alternate)")
	plt.show()


if __name__ == '__main__':
	float_formatter = "{:.3f}".format
	np.set_printoptions(formatter={'float_kind':float_formatter})

	run_indexability_experiment()
	run_limit_cycle_experiment()
	run_reward_vs_horizon()
	run_reward_vs_gamma()
	run_reward_vs_mixing()