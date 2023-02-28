import numpy as np 

# incrementally create a Q that respects:
# - Q is increasing in s wrt to a and k
# - Q is increasing in a wrt to s and k
def fastQ_strict(S,A):
	
	Q = np.zeros((S,A,S))
	Q[:,:,0] = 1

	means = np.linspace(0,1,S+1)[1:]-1/(S*2)
	sd = 1/(S*2*3) 

	# Step 1: Roll S numbers and sort to create top-right entries
	draw = np.sort(np.random.normal(means, [sd]*S))
	for s in range(S):
		Q[s,0,S-1] = min(1, max(0, draw[s]))


	# Step 2: set the next action for each state, using worse action/state probs as lower bounds
	for a in range(1,A):
		for s in range(S):

			lb = 0
			ub = 1
			# just check above action
			if s == 0:
				lb = Q[s,a-1,S-1]
			# check above action and above state
			else:
				lb = max(Q[s,a-1,S-1],Q[s-1,a,S-1]) 

			entry = np.random.rand() * (ub-lb) + lb
			Q[s,a,S-1] = entry


	# Step 3: loop in reverse over the inner states and do the same thing
	for k in list(range(1,S-1))[::-1]:
		for a in range(0,A):
			for s in range(S):
				lb = 0
				ub = 1

				# just check right k
				if s == 0 and a == 0:
					lb = Q[s,a,k+1]
				# check right k and above state
				elif a == 0:
					lb = max( Q[s-1,a,k], Q[s,a,k+1])
				# check right k and above action
				elif s == 0:
					lb = max( Q[s,a-1,k], Q[s,a,k+1])
				# else check them all
				else:
					lb = max(Q[s,a-1,k],Q[s-1,a,k],Q[s,a,k+1]) 

				entry = np.random.rand() * (ub-lb) + lb
				Q[s,a,k] = entry		

	return Q

def make_T_from_q(Q):
	
	T = np.zeros(Q.shape)

	T[:,:,-1] = Q[:,:,-1]
	for i in list(range(Q.shape[2]-1))[::-1]:
		T[:,:,i] = Q[:,:,i] - Q[:,:,i+1]
	
	return T


# Check that a T respects:
# - Q is increasing in s wrt to a and k
# - Q is increasing in a wrt to s and k
def check_T_strict(T):
	S = T.shape[0]
	A = T.shape[1]
	Q = np.zeros((S,A,S))

	for s in range(S):
		for a in range(A):
			for k in range(S):
				Q[s,a,k] = T[s,a,k:].sum()


	# Covers the p11 > p01
	for k in range(S):
		for a in range(A):

			non_decreasing_in_S = True
			previous_value = 0
			for s in range(S):
				non_decreasing_in_S = Q[s,a,k] >= previous_value
				if not non_decreasing_in_S:
					return False
				previous_value = Q[s,a,k]

	# Ensure that action effects 
	# does this check preclude the first? No.
	# I think this covers p11a > p11p but need to verify
	for s in range(S):
		for k in range(S):

			non_decreasing_in_a = True
			previous_value = 0
			for a in range(A):
				non_decreasing_in_a = Q[s,a,k] >= previous_value
				if not non_decreasing_in_a:
					return False
				previous_value = Q[s,a,k]


	return True


# Check that a T respects:
# - Q is increasing in s wrt to a and k
def check_T_puterman(T):
	S = T.shape[0]
	A = T.shape[1]
	Q = np.zeros((S,A,S))

	for s in range(S):
		for a in range(A):
			for k in range(S):
				Q[s,a,k] = T[s,a,k:].sum()


	# Covers the p11 > p01
	for k in range(S):
		for a in range(A):

			non_decreasing_in_S = True
			previous_value = 0
			for s in range(S):
				non_decreasing_in_S = Q[s,a,k] >= previous_value
				if not non_decreasing_in_S:
					return False
				previous_value = Q[s,a,k]

	return True

def no_check(T):
	return True

def random_T(S,A,check_function=check_T_strict):

	T = None
	T_passed_check = False
	count_check_failures = 0
	while (not T_passed_check):
		count_check_failures += 1
		if count_check_failures %1000 == 0:
			print('count_check_failures:',count_check_failures)
		T = np.random.dirichlet(np.ones(S), size=(S,A))
		T_passed_check = check_function(T)

	return T





# Distribution used to generate Fig 3(a) in the paper
def get_full_random_experiment(N, S, A, REWARD_BOUND):

	T = np.zeros((N,S,A,S))
	for i in range(N):
		T[i] = random_T(S,A,check_function=no_check)

	R = np.sort(np.random.rand(N, S), axis=1)*REWARD_BOUND
	# R = np.array([np.arange(S) for _ in range(N)])


	C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

	B = N/8


	return T, R, C, B


def eng1_T(S,A,check_function=check_T_strict):

	T = np.zeros((S,A,S))

	prior_weight = 5
	for i in range(S):
		for j in range(A):

			prior = np.ones(S)
			add_vector = np.zeros(S)
			add_vector[i]+= prior_weight*abs(j-A)
			prior += add_vector
			T[i,j] = np.random.dirichlet(prior)


	return T

# Distribution used to generate Fig 3(b) in the paper
def get_eng1_experiment(N, S, A, B, REWARD_BOUND):


	A = 4
	T = np.zeros((N,S,A,S))
	for i in range(N):
		T[i] = eng1_T(S,A,check_function=no_check)


	R = np.array([np.arange(S) for _ in range(N)])



	#Source for SampleLam 3(b)
	# N = 1000
	C = np.array([0, 1, 5, 25])
	B = 2


	return T, R, C, B





def get_greedy(S,A,check_function=check_T_strict):

	T = np.zeros((S,A,S))

	must_act_to_move = np.zeros(S)
	must_act_to_move[1] = 1

	s = 0
	# First action must be 1 else or you get locked in the dead state
	T[s,1] = must_act_to_move
	for a in [0]+list(range(2,A)):
		must_act_to_move = np.zeros(S)
		must_act_to_move[-1] = 1
		T[s,a] = must_act_to_move

	for s in range(1,S):
		for a in range(A):
			if s+1 == a:
				increasing_action_cost = np.zeros(S)
				increasing_action_cost[a] = 1
				T[s,a] = increasing_action_cost

			else:
				must_act_to_move = np.zeros(S)
				must_act_to_move[-1] = 1
				T[s,a] = must_act_to_move

	return T


def get_reliable(S,A,check_function=check_T_strict):

	T = np.zeros((S,A,S))

	must_act_to_move = np.zeros(S)
	must_act_to_move[1] = 1

	s = 0
	# First action must be 1 else or you get locked in the dead state
	T[s,1] = must_act_to_move
	for a in [0]+list(range(2,A)):
		must_act_to_move = np.zeros(S)
		must_act_to_move[-1] = 1
		T[s,a] = must_act_to_move


	# after that, all actions except 0 keep you locked in state 1
	for s in range(1,S):
		for a in range(A):
			if s == 1 and a != 0:
				always_stay = np.zeros(S)
				always_stay[1] = 1
				T[s,a] = always_stay
			else:
				always_stay = np.zeros(S)
				always_stay[-1] = 1
				T[s,a] = always_stay

	return T


def get_lam0(S,A):

	T = np.zeros((S,A,S))

	w = 2000

	stay_in_0_prior = np.ones(S)
	stay_in_0_prior[0] = w
	for s in range(S):
		for a in range(A):
			T[s,a] = np.random.dirichlet(stay_in_0_prior)

	# T[:,:,0] = 1

	return T

# Greedy/Reliable/Easy experiment from the paper
def get_eng11_experiment(N, A, percent_greedy, percent_lam0, REWARD_BOUND):
	print((N, A, percent_greedy, percent_lam0, REWARD_BOUND))

	S = A+1
	T = np.zeros((N,S,A,S))

	num_greedy = int(N*percent_greedy)
	for i in range(num_greedy):
		#cliff, no ladder
		T[i] = get_greedy(S,A,check_function=no_check)
		# print("getting nonrecov")
	num_lam0 = int(N*percent_lam0)
	for i in range(num_greedy, num_lam0+num_greedy):
		#cliff with ladder
		T[i] = get_lam0(S,A)
		# print("getting good on their own")
	for i in range(num_lam0+num_greedy, N):
		T[i] = get_reliable(S,A,check_function=no_check)

	assert (np.abs(T.sum(axis=-1) - 1) < 1e-3).all()

	R = np.zeros((N,S))
	R[:,:-1] = 1 + 1.01*np.arange(S-1)
	# R looks like [1, 2, 3, ..., S-1, 0]

	C = np.arange(A)

	return T, R, C






def get_state_translators(num_adherence_states, IP_length):

 state_to_tup = {}
 tup_to_state = {}

 # include a "day" for the CP (will recur on itself)
 uniq_id = 0
 for i in range(IP_length+2):
  for j in range(num_adherence_states):
   s = uniq_id
   t = (i,j)
   state_to_tup[s] = t
   tup_to_state[t] = s
   uniq_id += 1

 return state_to_tup, tup_to_state





def get_patient(num_adherence_states, IP_length, patient_type, tup_to_state, REWARD_BOUND):

 A = 5
 S = num_adherence_states * (IP_length+2)
 T = np.zeros((A,S,S))
 R = np.zeros(S)

 R_sub = np.arange(num_adherence_states)/(num_adherence_states-1)*REWARD_BOUND

 # In the IP phase, patients are more susceptible to actions,
 # each of which tend to boost adherence 


 baseline_prob = None
 action_units = None
 state_unit = None
 adh_state_upper_bound = None


 p_go_to_failure = np.random.rand()*0.25
 p_leave_failure = p_go_to_failure


 # Set basic state and action effects for the intensive phase
 if patient_type == 'good':
  baseline_prob = np.random.dirichlet([1,1,1000])
  action_units = np.eye(3)*np.random.rand()*0.00
  state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.00
  adh_state_upper_bound = num_adherence_states-1


 elif patient_type == 'unresponsive':
  baseline_prob = np.random.dirichlet([1000,10,10])
  action_units = np.eye(3)*np.random.rand()*0.01
  state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.01
  adh_state_upper_bound = num_adherence_states-1

 elif patient_type == 'responsive':
  baseline_prob = np.random.dirichlet([20,20,20])
  # action_unit = np.ones(3)*np.random.rand()*1
  action_units = [
      [5.0, 1.0, 1.0],
      [1.0, 5.0, 1.0],
      [1.0, 1.0, 5.0]
      ]
  state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.1

  # sample a random upper bound for their adherence state so that patients have different
  # expected rewards when you always act on them
  # responsive patients will have random upper bound for adherence state between 50th percentile and 100th percentile state
  # adh_state_upper_bound = (num_adherence_states-1)//2
  # adh_state_upper_bound = int(round((num_adherence_states-1)*.8))
  adh_state_upper_bound = int(round((num_adherence_states-1) * (np.random.rand()*.5 + 0.5)))


 elif patient_type == 'responsive_dropout':
  baseline_prob = np.random.dirichlet([20,20,20])
  action_units = [
      [20.0, 1.0, 1.0],
      [1.0, 20.0, 1.0],
      [1.0, 1.0, 20.0]
      ]
  state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.1

  # sample a random upper bound for their adherence state so that patients have different
  # expected rewards when you always act on them
  # responsive patients will have random upper bound for adherence state between 50th percentile and 100th percentile state
  adh_state_upper_bound = num_adherence_states-1
  # adh_state_upper_bound = int(round((num_adherence_states-1) * (np.random.rand()*.5 + 0.5)))

 # going to use this to determine how much to add to each entry of the baseline probs
 probability_split_assigner = np.linspace(0,2,num_adherence_states)
 for day in range(IP_length+1):

  # Set basic state and action effects for the continuation phase
  # once we reach CP, actions have less effect, but state has more effect
  if day == IP_length:

   if patient_type == 'good':
    # these will sustain themselves no matter waht
    # baseline_prob = np.random.dirichlet([10,10,80])
    action_units = np.eye(3)*np.random.rand()*0.01
    state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.01

   elif patient_type == 'unresponsive':
    # these will tend to fail no matter what
    # baseline_prob = np.random.dirichlet([80,10,10])
    action_units = np.eye(3)*np.random.rand()*0.01

    state_unit_vector = np.ones(num_adherence_states)*np.random.rand()*0.01

   elif patient_type == 'responsive':
    # in the IP phase, responsive patients need to be called
    # to sustian, or need a house visit to increase their adherence
    baseline_prob = np.array([0.0,0.0,0.0])
    action_units = [
        [100.0, 1.0, 1.0],
        [1.0, 100.0, 1.0],
        [1.0, 100.0, 1.0]
        ]

    # low states will have holding power, while higher states just need small actions to maintain
    state_unit_vector = np.linspace(10, 0, num_adherence_states)
   
   elif  patient_type == 'responsive_dropout':

    baseline_prob = np.array([0.0,0.0,0.0])
    action_units = [
        [100.0, 1.0, 1.0],
        [1.0, 100.0, 1.0],
        [1.0, 100.0, 1.0]
        ]
    state_unit_vector = np.zeros(num_adherence_states)


  for a in range(A-2):
   for adh_today in range(num_adherence_states):

    current_prob = np.array(baseline_prob)

    # for each of the first three actions, directly add action_unit
    # to the prob of going down, staying or going up

    current_prob += action_units[a]

    # now, add weight to the probability of going down, staying, or going up
    # based on what state you are in.
    # The higher the state you are in, the more weight we add to prob going up
    split_share = probability_split_assigner[adh_today]
    split_share_base_ind = int(split_share)

    split_share_up = split_share - int(split_share) 
    split_share_base = 1 - split_share_up

    current_prob[split_share_base_ind] += split_share_base*state_unit_vector[adh_today]
    if split_share_base_ind+1 < len(current_prob):
     current_prob[split_share_base_ind+1] += split_share_up*state_unit_vector[adh_today]


    # renormalize
    current_prob /= current_prob.sum()
    t = (day, adh_today)
    s = tup_to_state[t]

    R[s] = R_sub[adh_today]



    # allow the actions to have their normal effect up to the adherence upper bound
    if adh_today <= adh_state_upper_bound:

     # if we are at the adh_upper_bound, but lower than the num_adherence_states,
     # give some small prob of still going up
     if adh_today == adh_state_upper_bound and adh_state_upper_bound < num_adherence_states - 1:

      # combine prob(going up) with prob(stay)
      current_prob[1] += current_prob[2]

      # give a small prob(going up)
      current_prob[2] = np.random.rand()*0.1

      # renormalize
      current_prob /= current_prob.sum()


     for prob_ind,adh_shift in enumerate([-1, 0, 1]):
      adh_next = min(max(0, adh_today+adh_shift), num_adherence_states-1)

      next_day = day+1
      if day == IP_length:
       next_day = day

      t = (next_day, adh_next)
      sp = tup_to_state[t]

      T[a,s,sp] += current_prob[prob_ind]



    # once past the adh_state_upper_bound, patient adherence should decay back to the upper bound
    elif adh_today > adh_state_upper_bound:

     prob = np.random.dirichlet([100, 1, 1])
     # prob = np.array([.9,.05,.05])
     for prob_ind,adh_shift in enumerate([-1, 0, 1]):
      adh_next = min(max(0, adh_today+adh_shift), num_adherence_states-1)

      next_day = day+1
      if day == IP_length:
       next_day = day

      t = (next_day, adh_next)
      sp = tup_to_state[t]

      T[a,s,sp] += prob[prob_ind]


    # deal with transfer to failure phase for dropouts
    if day == IP_length and patient_type == 'responsive_dropout':
     p_top_state = 0.0
     p_bottom_state = 1 - p_top_state

     # get probs for cotinuation phase
     indices = [tup_to_state[(day, adh_next)] for adh_next in range(num_adherence_states)]
     probs = np.array(T[a,s,indices])

     # probs for staying in continutation phase
     probs *= (1 - p_go_to_failure)
     for adh_next in range(num_adherence_states):
      t = (day, adh_next)
      sp = tup_to_state[t]

      T[a,s,sp] = probs[adh_next] 

     # probs for going to failure phase
     t = (day+1, num_adherence_states-1)
     sp = tup_to_state[t]
     T[a,s,sp] = p_top_state*p_go_to_failure

     t = (day+1, 0)
     sp = tup_to_state[t]
     T[a,s,sp] = p_bottom_state*p_go_to_failure



 # savior actions should have higher prob of returning patient to highest states

 # This parameter tunes the strength of savior actions
 w = None
 if patient_type == 'good':
  w = 500
 elif patient_type == 'unresponsive':
  w = 10
 elif patient_type == 'responsive' or patient_type == 'responsive_dropout':
  w = 25


 for a in range(A-2, A):
  for day in range(IP_length+1):
   for adh_today in range(num_adherence_states):
    

    prior = [1 for i in range(adh_today)] + [w for i in range(adh_today, num_adherence_states)]
    
    prior = np.ones(num_adherence_states)
    prior[-1] = w
    if a == A-1:
     # give the ESCALATE action even higher chance of returning to top state
     prior[-1] *= 5

    prior = np.cumsum(prior)
    probs = np.random.dirichlet(prior)

    t = (day, adh_today)
    s = tup_to_state[t]

    R[s] = R_sub[adh_today]

    # deal with transfer to failure phase for dropouts
    if day == IP_length and patient_type == 'responsive_dropout':
     p_top_state = 1.0
     p_bottom_state = 1 - p_top_state


     # probs for staying in continutation phase
     probs *= (1 - p_go_to_failure)

     for adh_next in range(num_adherence_states):
      t = (day, adh_next)
      sp = tup_to_state[t]

      T[a,s,sp] = probs[adh_next] 

     # probs for going to failure phase
     t = (day+1, num_adherence_states-1)
     sp = tup_to_state[t]
     T[a,s,sp] = p_top_state*p_go_to_failure

     t = (day+1, 0)
     sp = tup_to_state[t]
     T[a,s,sp] = p_bottom_state*p_go_to_failure

    # for non-dropouts, guaranteed to stay in CP phase
    else:
     for adh_next in range(num_adherence_states):
      next_day = day+1
      if day == IP_length:
       next_day = day

      t = (next_day, adh_next)
      sp = tup_to_state[t]

      T[a,s,sp] = probs[adh_next] 


 # deal with the failure phase
 # in the failure phase, you can spend escalate actions to recover patients
 # to the CP with p_go_to_failure probability
 day = IP_length+1

 # non-escalate actions have no effect
 for a in range(A-2):
  for adh_today in range(num_adherence_states):

   # current state
   t = (day, adh_today)
   s = tup_to_state[t]

   # set reward
   R[s] = R_sub[adh_today]

   # next state
   t = (day, 0)
   sp = tup_to_state[t]
   # print(a,s,sp)
   # always go to worst state immediately on a non-savior action during FP
   T[a,s,sp] = 1

 for a in range(A-2, A):
  for adh_today in range(num_adherence_states):

   # current state
   t = (day, adh_today)
   s = tup_to_state[t]

   # set reward
   R[s] = R_sub[adh_today]
  

   # probs for staying in failure phase
   p_top_state = 1.0
   p_bottom_state = 1 - p_top_state

   t = (day, num_adherence_states-1)
   sp = tup_to_state[t]
   T[a,s,sp] = p_top_state*(1 - p_leave_failure)

   t = (day, 0)
   sp = tup_to_state[t]
   T[a,s,sp] = p_bottom_state*(1 - p_leave_failure)


   # probs for returning to CP
   p_top_state = 1.0
   p_bottom_state = 1 - p_top_state

   t = (day-1, num_adherence_states-1)
   sp = tup_to_state[t]
   T[a,s,sp] = p_top_state*(p_leave_failure)

   t = (day-1, 0)
   sp = tup_to_state[t]
   T[a,s,sp] = p_bottom_state*(p_leave_failure)

   


 return T, R



# healthcare experiment - puterman condition?
# every action makes you better off than the last but at increasingly greater cost
# state:
# (adherence, time_step, treatment_phase)
def get_healthcare_experiment(N, num_adherence_states, IP_length, percent_good_patients, 
		percent_unresponsive_patients, percent_responsive_patients, B, REWARD_BOUND):

	# n_states = num_adherence_states * (IP_length+1)

 S = num_adherence_states * (IP_length+2)
 A = 5
 # A = 4
 T = np.zeros((N,A,S,S))
 R = np.zeros((N,S))

 # print('size of T',T.shape)
 # print('Generating matrices...')

 state_to_tup, tup_to_state = get_state_translators(num_adherence_states, IP_length)

 num_good = int(N*percent_good_patients)
 for i in range(num_good):
  T[i], R[i] = get_patient(num_adherence_states, IP_length, 'good', tup_to_state, REWARD_BOUND)
  # print("good")
 num_unresponsive = int(N*percent_unresponsive_patients)
 for i in range(num_good, num_unresponsive+num_good):
  T[i], R[i] = get_patient(num_adherence_states, IP_length, 'unresponsive', tup_to_state, REWARD_BOUND)
  # print("unresponsive")
 num_responsive = int(N*percent_responsive_patients)
 for i in range(num_unresponsive+num_good, num_unresponsive+num_good+num_responsive):
  T[i], R[i] = get_patient(num_adherence_states, IP_length, 'responsive', tup_to_state, REWARD_BOUND)
  # print("resopnsive")
 for i in range(num_unresponsive+num_good+num_responsive, N):
  T[i], R[i] = get_patient(num_adherence_states, IP_length, 'responsive_dropout', tup_to_state, REWARD_BOUND)
  # print("resopnsive dropout")

 # Go from N,A,S,S to N,S,A,S
 T = np.swapaxes(T,1,2)

 # actions [nothing, call, take appointment, visit with specialist, ESCALATE]
 # budget will determine how many calls we can make
 # can always only do 2 specialist appointments per day or one escalation
 # C = np.array([0, 1, 2, B, B])
 C = np.array([0, 1, 2, B/2, B])
 # C = np.array([0, 1, 2, 8, 8])

 # starting state in tuple form is (day=0, adherence=max(adherence))
 st = (0, num_adherence_states - 1)
 ss = tup_to_state[st]
 start_state = np.ones(N)*ss

 print("Done generating matrices")

 return T, R, C, start_state




def get_strict_random_experiment(N, S, A, REWARD_BOUND):

	T = np.zeros((N,S,A,S))
	for i in range(N):
		T[i] = random_T(S,A,check_function=check_T_strict)

	# R = np.sort(np.random.rand(N, S), axis=1)*REWARD_BOUND
	R = np.array([np.arange(S) for _ in range(N)])

	C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

	B = N/2

	return T, R, C, B

def get_strict_fast_experiment(N, S, A, REWARD_BOUND):

	T = np.zeros((N,S,A,S))
	for i in range(N):
		Q = fastQ_strict(S,A)
		T[i] = make_T_from_q(Q)
		if not check_T_strict(T[i]):
			raise ValueError("T not valid")


	R = np.array([np.arange(S) for _ in range(N)])

	C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

	B = N/2

	return T, R, C, B


def get_puterman_random_experiment(N, S, A, REWARD_BOUND):

	T = np.zeros((N,S,A,S))
	for i in range(N):
		T[i] = random_T(S,A,check_function=check_T_puterman)


	# R = np.array([np.arange(S) for _ in range(N)])

	R = np.array([np.arange(S) for _ in range(N)])

	C = np.concatenate([[0], np.sort(np.random.rand(A-1))])

	# C = np.array([0, 1, 5, 25])


	B = N/4
	# epsilon = 1e-3
	# C[-1] = B-epsilon
	

	return T, R, C, B



def get_adverse_for_lagrangian_one_class_5(N, leakS, leakR, leakD):
	"""
	Get the 5 state homogenous RMAB
	States:
	- 0: START-1
	- 1: ENGAGED-1
	- 2: START-2
	- 3: ENGAGED-2
	- 4: DROPOUT-1
	"""
	N_CLS, N_STATES, N_ACTIONS = N, 5, 2
	P = np.zeros((N_CLS, N_STATES, N_ACTIONS, N_STATES))

	# START-1
	P[:,0,0] = np.array([ 0, leakS, 0, 0, 1-leakS])
	P[:,0,1] = np.array([ 0, 1, 0, 0, 0])

	# ENGAGED-1
	P[:,1,0] = np.array([ 0, 0, 0, 0, 1])
	P[:,1,1] = np.array([ 0, 1-leakR, 0, 0, leakR])

	# START-2
	P[:,2,0] = np.array([ 0, 0, 0, leakS, 1-leakS])
	P[:,2,1] = np.array([ 0, 0, 0, 1, 0])

	# ENGAGED-2
	P[:,3,0] = np.array([ 0, 0, 0, 0, 1])
	P[:,3,1] = np.array([ 0, 0, 0, 0, 1])

	# DROPOUT
	P[:,4,0] = np.array([ leakD/2, 0, leakD/2, 0, 1-leakD])
	P[:,4,1] = np.array([ leakD/2, 0, leakD/2, 0, 1-leakD])

	R = np.zeros((N_CLS, N_STATES))
	R[:] = np.array([0, 0.99, 0, 1, 0])

	C = np.array([0,1])

	B = N/2

	start_state = np.zeros(N)
	start_state[int(N/2):] = 2
	
	return P, R, C, B, start_state

def get_adverse_for_lagrangian_one_class_6(N):
	"""
	Get the 6 state homogenous RMAB
	States:
	- 0: START-1
	- 1: ENGAGED-1
	- 2: DROPOUT-1
	- 3: START-2
	- 4: ENGAGED-2
	- 5: DROPOUT-2
	"""
	N_CLS, N_STATES, N_ACTIONS = N, 6, 2
	P = np.zeros((N_CLS, N_STATES, N_ACTIONS, N_STATES))
	leak = 0.1

	# START-1
	P[:,0,0] = np.array([ 0, 0, 1-leak, 0, 0, leak])
	P[:,0,1] = np.array([ 0, 1-leak, 0, 0, 0, leak])

	# ENGAGED-1
	P[:,1,0] = np.array([ 0, 0, 1-leak, 0, 0, leak])
	P[:,1,1] = np.array([ 0, 1, 0, 0, 0, 0])

	# DROPOUT-1
	P[:,2,0] = np.array([ 0, 0, 1-leak, 0, 0, leak])
	P[:,2,1] = np.array([ 0, 0, 1-leak, 0, 0, leak])

	# START-2
	P[:,3,0] = np.array([ 0, 0, leak, 0, 0, 1-leak])
	P[:,3,1] = np.array([ 0, 0, leak, 0, 1-leak, 0])

	# ENGAGED-2
	P[:,4,0] = np.array([ 0, 0, leak, 0, 0, 1-leak])
	P[:,4,1] = np.array([ 0, 0, leak, 0, 0, 1-leak])

	# DROPOUT-2
	P[:,5,0] = np.array([ 0, 0, leak, 0, 0, 1-leak])
	P[:,5,1] = np.array([ 0, 0, leak, 0, 0, 1-leak])

	R = np.zeros((N_CLS, N_STATES))
	R[:] = np.array([0, 0.99, 0, 0, 1, 0])

	C = np.array([0,1])

	B = N/2

	start_state = np.zeros(N)
	start_state[int(N/2):] = 2
	
	return P, R, C, B, start_state


def get_adverse_for_lagrangian_two_class(N):
	N_CLS, N_STATES, N_ACTIONS = N, 3, 2
	P = np.zeros((N_CLS, N_STATES, N_ACTIONS, N_STATES))

	# Both
	P[:,:,:] = np.array([0,0,1])
	P[:,2,:] = np.array([0,0,1])
	
	# Greedy
	P[:int(N_CLS/2),0,1] = np.array([0,1,0])

	# Reliable
	P[int(N_CLS/2):,0,1] = np.array([0,1,0])
	P[int(N_CLS/2):,1,1] = np.array([0,1,0])

	R = np.zeros((N_CLS, N_STATES))
	R[:int(N_CLS/2),:] = np.array([0,1,0]).reshape((1,N_STATES))
	R[int(N_CLS/2):,:] = np.array([0,0.5,0]).reshape((1,N_STATES))

	C = np.array([0,1])

	B = N/2

	start_state = np.zeros(N)
	
	return P, R, C, B, start_state


def get_adverse_for_lagrangian(N, leakS = 0.01, leakR = 0.01, leakD = 0.01):
	return get_adverse_for_lagrangian_one_class_5(N, leakS, leakR, leakD)


def get_adverse_for_mean_field(N):
	"""
	States:
	s[0] to s[9]; dropout == s[4]
	"""
	N_CLS, N_STATES, N_ACTIONS = N, 10, 2
	P = np.zeros((N_CLS, N_STATES, N_ACTIONS, N_STATES))


	# default, go to dropout = s5 = s[4]
	P[:,:,:,4] = 1

	# s[0]
	P[:,0,1] = np.array([0, 1, 0, 0, 0, 0, 0, 0, 0, 0])
	P[:,0,0] = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
	
	# s[1]
	P[:,1,0] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0])
	# s[2]
	P[:,2,1] = np.array([0, 0, 0, 0.5, 0.5, 0, 0, 0, 0, 0])
	# s[3]
	P[:,3,1] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
	# s[4]
	pass  # dropout
	# s[5]
	P[:,5,1] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])

	# s[6]
	P[:,6,1] = np.array([0, 0, 0, 0, 0, 0, 0, 1, 0, 0])
	# s[7]
	P[:,7,1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 1, 0])
	# s[8]
	P[:,8,1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])
	# s[9]
	P[:,9,1] = np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 1])

	R = np.zeros((N_CLS, N_STATES))
	R[:] = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0.99])

	C = np.array([0,1])

	B = int(N/3)

	start_state = np.zeros(N)
	start_state[int(2*N/3):] = 6

	print(start_state)
	
	return P, R, C, B, start_state