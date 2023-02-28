import multiprocessing
import numpy as np
import pandas as pd
import time

import utils
from itertools import product

import lp_methods
import lp_methods_gurobi
import simulation_environments

import os
import argparse
import tqdm 
import itertools

import mdptoolbox
from numba import jit

import matplotlib.pyplot as plt

index_policies = [19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40]

percent_good_patients = 0
percent_unresponsive_patients = 0
percent_responsive_patients = 0

def plot_bounds(s_ub, i_ub, s_lb, i_lb, num_grid_points_per_process, grid_points,
                V_slope_grid_debug=None, V_intercept_grid_debug=None,
                debug_grid=None):

    state = 0#np.random.randint(0,s_ub.shape[2]+1)

    N = s_ub.shape[0]

    for i in range(s_ub.shape[0]):
        if num_grid_points_per_process[i] == 3 or True:
            print(num_grid_points_per_process[i])
            print(s_ub[i,:,state], i_ub[i,:,state])
            print(s_lb[i,:,state], i_lb[i,:,state])
            print(grid_points[i,:])
            title = ""
            print (percent_good_patients*N, i , (percent_good_patients+percent_unresponsive_patients)*N)
            if i < percent_good_patients*N:
                title = "good"
            elif percent_good_patients*N < i and i < (percent_good_patients+percent_unresponsive_patients)*N:
                title = "unresponsive"
            elif (percent_good_patients+percent_unresponsive_patients)*N < i and i < (percent_good_patients+percent_unresponsive_patients+percent_responsive_patients)*N:
                title = "responsive"
            else:
                title = 'responsive migrant'
            plt.title(title)
            for j in range(num_grid_points_per_process[i]):
                u = 1
                if j < num_grid_points_per_process[i] - 1:
                    u = grid_points[i,j+1]
                x = np.linspace(grid_points[i,j], u, 100)

                plt.plot(x, s_ub[i,j,state]*x + i_ub[i,j,state], c='r')
                plt.plot(x, s_lb[i,j,state]*x + i_lb[i,j,state], c='g')

            for j in range(debug_grid.shape[1]):
                u = 1
                print(i,j)
                if j < debug_grid.shape[1] - 1:
                    u = debug_grid[i,j+1]
                x = np.linspace(debug_grid[i,j], u, 100)

                plt.plot(x, V_slope_grid_debug[i,j,state]*x + V_intercept_grid_debug[i,j,state], c='b')


            plt.show()
            # if i > 10:
            #     1/0
    1/0


def takeAction(current_states, T, actions, random_stream):

    N=len(current_states)

    ###### Get next state
    next_states=np.zeros(current_states.shape)
    for i in range(N):

        current_state=int(current_states[i])
        next_state=np.argmax(random_stream.multinomial(1, T[i, current_state, int(actions[i]), :]))
        next_states[i]=next_state

    return next_states


def getActions(N, T, R, C, B, k, valid_action_combinations=None, combined_state_dict=None, current_state=None, 
                optimal_policy=None, policy_option=0, days_remaining=None, simulation_length=None, 
                gamma=0.95, mean_field_clusters=0, indexes=None, output_data=None, blam_data=None):

   
    if policy_option==0:
        ################## Nobody
        return np.zeros(N)

    elif policy_option==1:
        ################## Everybody
        return np.ones(N)

    elif policy_option==2:
        ################## Random
        # Randomly pick from all valid options
        choices = np.arange(valid_action_combinations.shape[0])
        choice = np.random.choice(choices)
        return valid_action_combinations[choice]


    # Round robin 1
    elif policy_option==5:
        actions = np.zeros(N)
        num_feasible = int(B/C[1])
        last_proc_acted_on = output_data['last_proc_acted_on_rr']
        ind = 0
        for i in range(last_proc_acted_on+1, last_proc_acted_on+1 + num_feasible):
            ind = i%N
            actions[ind] = 1

        output_data['last_proc_acted_on_rr'] = ind
        return actions


    # Round robin 2
    elif policy_option==6:
        actions = np.zeros(N)
        num_feasible = int(B/C[2])
        last_proc_acted_on = output_data['last_proc_acted_on_rr']
        ind = 0
        for i in range(last_proc_acted_on+1, last_proc_acted_on+1 + num_feasible):
            ind = i%N
            actions[ind] = 2

        output_data['last_proc_acted_on_rr'] = ind
        return actions

    # Round robin E
    elif policy_option==7:

        actions = np.zeros(N)
        num_feasible = int(B/C[-1])
        last_proc_acted_on = output_data['last_proc_acted_on_rr']
        ind = 0
        for i in range(last_proc_acted_on+1, last_proc_acted_on+1 + num_feasible):
            ind = i%N
            actions[ind] = len(C) - 1

        output_data['last_proc_acted_on_rr'] = ind
        return actions


    # Mean-Field Finite Time
    elif policy_option in [50, 51, 52, 53, 54]:
        """
        50: cluster with provided args.mean_field_clusters
        51: no clustering
        52: cluster 20
        53: cluster 50
        54: cluster 100
        """
        CLUSTER_COUNTS = {
            50: mean_field_clusters, 
            51: 0,
            52: 20,
            53: 50,
            54: 100
        }
        N_CLUSTERS = CLUSTER_COUNTS[policy_option]

        if days_remaining + 1 == simulation_length: 
            assert output_data['mf_clusters'][policy_option] is None
        
        # we may not tell mean-field the number of remaining days to make the game fair
        # ... also, make it do clustering each time to get the real-life estimate of time        
        # replace days_remaining with simulation_length

        # tell the remaining days below
        actions, output_data['mf_clusters'][policy_option] = lp_methods.mean_field_wrapper_jackson(
            T, R, C, B, current_state, days_remaining, gamma, N_CLUSTERS, output_data['mf_clusters'][policy_option])

        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B + 1e-3: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")

        return actions



    # Hawkins
    elif policy_option==21:

        actions = np.zeros(N)

        lambda_lim = R.max()/(C[C>0].min()*(1-gamma))

        indexes = np.zeros((N, C.shape[0], T.shape[1]))
        L_vals, lambda_val = lp_methods_gurobi.hawkins(T, R, C, B, current_state, lambda_lim=lambda_lim, gamma=gamma)

        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] - lambda_val*C[a] + gamma*L_vals[i].dot(T[i,s,a])
        output_data['hawkins_lambda'].append(lambda_val)

        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]

        decision_matrix = lp_methods_gurobi.action_knapsack(indexes_per_state, C, B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: 
            print("budget")
            print(B)
            print("Cost")
            print(C)
            print("ACTIONS")
            print(actions)
            raise ValueError("Over budget")

        return actions



    # No cost value funcs
    elif policy_option==24:

        actions = np.zeros(N)
        
        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        decision_matrix = lp_methods_gurobi.action_knapsack(indexes_per_state, C, B)

        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

        return actions


    # SampleLam
    elif policy_option == 27:

        N_samples_samplelam = int(output_data['nsamples_samplelam'])
        # print("N_samples_samplelam")
        # print(N_samples_samplelam)

        # Compute necessary samples based on delta and upper reward bound
        # ub = R.max()/(1-gamma)

        # N_samples_samplelam = int(nsamples_samplelam)

        indexes = np.zeros((N, C.shape[0], T.shape[1]))
        V_all = np.zeros((N, T.shape[1]))


        sub_sample = np.random.choice(np.arange(N), N_samples_samplelam, replace=False).tolist()

        B_unit = B/N

        sampled_lam_list = np.zeros(N_samples_samplelam)

        # print('B_unit',B_unit)
        
        for i in range(N_samples_samplelam):
            proc_ind = sub_sample[i]
            V_vals, lambda_val, _ = lp_methods_gurobi.sample_lam(T[[proc_ind]], R[[proc_ind]], C, B_unit, current_state[[proc_ind]], gamma=gamma)
            sampled_lam_list[i] = lambda_val

        output_data['lambdas_samplelam'].append(sampled_lam_list)


        lambda_val = sampled_lam_list.mean()
        # print('lambda_val')
        # print(lambda_val)
        # Value iteration on the remaining, using found lambda
        for i in range(N):
            # Go from S,A,S to A,S,S
            T_i = np.swapaxes(T[i],0,1)

            R_i = np.zeros(T_i.shape)
            for x in range(R_i.shape[0]):
                # subtract the lambda-weighted cost of acting
                R_i[x] -= lambda_val*C[x]
                for y in range(R_i.shape[1]):
                    R_i[x,:,y] += R[i]
   
            # print(T_i)
            # print(R_i)
            mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='fast', epsilon=output_data['mdp_epsilon'])
            mdp.run()

            V_all[i] = np.array(mdp.V)

        
        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] - lambda_val*C[a] + gamma*V_all[i].dot(T[i,s,a])


        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]


        decision_matrix = lp_methods_gurobi.action_knapsack(indexes_per_state, C, B)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")


        return actions


    # get_actions: BLam with grid search from the front
    # start with just the one aux variable and go from there
    elif policy_option in [33, 34, 35, 36, 37, 38, 39, 40]:

        
        V_slope_grid_ub = blam_data['V_slope_grid_ub']
        V_intercept_grid_ub = blam_data['V_intercept_grid_ub']

        V_slope_grid_lb = blam_data['V_slope_grid_lb']
        V_intercept_grid_lb = blam_data['V_intercept_grid_lb']
        
        bound_grid = blam_data['bound_grid']
        num_grid_points_per_process = blam_data['num_grid_points_per_process']

        grid_lengths = blam_data['grid_lengths']

        V_lam_final = np.zeros((N, T.shape[1]))



        indexes = np.zeros((N, C.shape[0], T.shape[1]))
        
        Q = np.zeros((N, C.shape[0], T.shape[1]))


        s = int(current_state[0])

        # get the slopes for the current state
        V_slopes_ub = np.zeros(bound_grid.shape)
        V_intercepts_ub = np.zeros(bound_grid.shape)

        V_slopes_lb = np.zeros(bound_grid.shape)
        V_intercepts_lb = np.zeros(bound_grid.shape)
        for i in range(N):
            for lambda_ind in range(num_grid_points_per_process[i]):
                V_slopes_ub[i, lambda_ind] = V_slope_grid_ub[i, lambda_ind, current_state[i]]
                V_intercepts_ub[i, lambda_ind] = V_intercept_grid_ub[i, lambda_ind, current_state[i]]

                V_slopes_lb[i, lambda_ind] = V_slope_grid_lb[i, lambda_ind, current_state[i]]
                V_intercepts_lb[i, lambda_ind] = V_intercept_grid_lb[i, lambda_ind, current_state[i]]


        # Want to sort the procs so that we are not unbounded to start
        # and so that the loosest bounds get pulled in first
        # Need a better way to pick the initial top k (sample lam?)

        #compute the cumulative slope of each process and bound out the ones that are smallest

        cumu_slopes = V_slopes_ub*grid_lengths
        cumu_slopes = cumu_slopes.sum(axis=1)
        slope_sort_inds = np.argsort(cumu_slopes)
        sorted_current_state = current_state[slope_sort_inds]

        # sort based on the last slope since it has biggest chance of hurting the bound

        final_slopes_per_process = np.array([V_slopes_ub[p, num_grid_points_per_process[p] - 1] for p in range(N)])
        

        final_slopes_per_process = np.array(final_slopes_per_process)

        slope_sort_inds = np.argsort(final_slopes_per_process)
        # print(slope_sort_inds)
        sorted_slopes = final_slopes_per_process[slope_sort_inds]
        # print(sorted_slopes)
        sorted_current_state = current_state[slope_sort_inds]



        lambda_loose_ub = R.max()/(C[C>0].min()*(1-gamma))


        B_unit = B/N
        B_sub = B_unit

        num_iters = N
        step_size_blam = int(np.sqrt(N))
        # step_size_blam = 1

        num_samples_used = 0

        lambda_lb = 0
        lambda_ub = np.inf
        output_data['blam_lbs'].append([])
        output_data['blam_ubs'].append([])
        output_data['blam_xvals'].append([])
        model_data = None


        # TODO:
        # update to make use of multiple slopes? Seems to work for now
        # worst case scenario is it underestimates where we should start 
        # and we take too many iterations
        # but should always converge because of the loose ub
        a = (sorted_slopes[::-1]*(-1)).cumsum()
        a = a - B/(1-gamma)
        a = (a <= 0).sum()
        ind_coeff_sums_lt_budget = N - a
        start_point = max(step_size_blam,ind_coeff_sums_lt_budget)



        prev_i = 0


        sorted_inds = slope_sort_inds[start_point:]


        
        for i in list(range(start_point, num_iters+1, step_size_blam))+[num_iters+1]:
            output_data['blam_xvals'][-1].append(i)

            # print("num",i)
            # Get upper bound
            V_vals, lambda_ub, _, model_data = lp_methods_gurobi.blam_gridsearch(T[slope_sort_inds[prev_i:i]],
             R[slope_sort_inds[prev_i:i]], V_slopes_ub, V_intercepts_ub, V_slopes_lb, V_intercepts_lb,
             num_grid_points_per_process, C, B, sorted_current_state[:i], sorted_inds, 'ub', 
             prev_model_data=model_data, lambda_lim=lambda_loose_ub, gamma=gamma)


            output_data['blam_ubs'][-1].append(lambda_ub)
            # print('lambda_ub',lambda_ub)

            # only want to pass in V_slopes when we first build the model
            V_slopes_ub = None
            V_intercepts_ub = None
            V_slopes_lb = None
            V_intercepts_lb = None
            num_grid_points_per_process = None


            # Get lower bound
            lam_coeff_adjust = 0
            V_vals, lambda_lb, _, model_data = lp_methods_gurobi.blam_gridsearch([], [], 
                V_slopes_ub, V_intercepts_ub, V_slopes_lb, V_intercepts_lb,
                num_grid_points_per_process, C, B, sorted_current_state[:i], sorted_inds, 'lb', 
                prev_model_data=model_data, gamma=gamma)


            output_data['blam_lbs'][-1].append(lambda_lb)
            # print('lambda_lb',lambda_lb)

            if lambda_ub - lambda_lb <= output_data['epsilon_blam']:
                num_samples_used = i
                break




            prev_i = i

        # L_vals, lambda_val = lp_methods_gurobi.hawkins(T, R, C, B, current_state, lambda_lim=lambda_loose_ub, gamma=gamma)
        # print('hawkins',lambda_val)

        lambda_val = (lambda_ub + lambda_lb) / 2
        output_data['blam_lambda_out'].append(lambda_val)
        # get V when lam = epsilon
        for i in range(N):
            # Go from S,A,S to A,S,S
            T_i = np.swapaxes(T[i],0,1)
            R_i = np.zeros(T_i.shape)
            for x in range(R_i.shape[0]):
                # subtract the lambda-weighted cost of acting
                R_i[x] -= lambda_val*C[x]
                for y in range(R_i.shape[1]):
                    R_i[x,:,y] += R[i]
   

            mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, initial_value=0, discount=gamma, stop_criterion='fast', epsilon=output_data['mdp_epsilon'])
            # mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, initial_value=0, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
            mdp.run()

            V_lam_final[i] = np.array(mdp.V)

        
        
        for i in range(N):
            for a in range(C.shape[0]):
                for s in range(T.shape[1]):
                    indexes[i,a,s] = R[i,s] - lambda_val*C[a] + gamma*V_lam_final[i].dot(T[i,s,a])


        actions = np.zeros(N)
        
        indexes_per_state = np.zeros((N, C.shape[0]))
        for i in range(N):
            s = current_state[i]
            indexes_per_state[i] = indexes[i,:,s]



        decision_matrix = lp_methods_gurobi.action_knapsack(indexes_per_state, C, B)
        actions = np.argmax(decision_matrix, axis=1)

        if not (decision_matrix.sum(axis=1) <= 1).all(): raise ValueError("More than one action per person")
        # print(actions)
        
        payment = 0
        for i in range(len(actions)):
            payment += C[actions[i]]
        if not payment <= B: raise ValueError("Over budget")

        return actions
    



def simulateAdherence(N, L, T, R, C, B, k, policy_option, optimal_policy=None, combined_state_dict=None,
                        action_logs={},
                        seedbase=None, savestring='trial', 
                        world_random_seed=None, verbose=False, file_root=None, gamma=0.95,
                        output_data=None, start_state=None, mean_field_clusters=None, simulation_length=None):


    world_random_stream = np.random.RandomState()
    world_random_stream.seed(world_random_seed)



    state_log=np.zeros((N,L), dtype=int)
    actions_record=np.zeros((N, L-1))

    if action_logs is not None:
        action_logs[policy_option] = []


    indexes = np.zeros((N,C.shape[0]))
    blam_data = {}

    # Round robin setups
    if policy_option in [5,6,7]:
        output_data['last_proc_acted_on_rr'] = N-1


    if policy_option in index_policies:

        lambdas = np.zeros((N,C.shape[0]))
        V = np.zeros((N,T.shape[1]))

        start = time.time()


        if policy_option == 21:
            pass


        # VfNc
        if policy_option == 24:

            start = time.time()
            indexes = np.zeros((N, C.shape[0], T.shape[1]))

            # time to: add variables, add constraints, optimize, extract variable values
            for i in range(N):
                # Go from S,A,S to A,S,S
                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] = R[i]

                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion='full', epsilon=output_data['mdp_epsilon'])
                mdp.run()

                V[i] = np.array(mdp.V)


            for i in range(N):
                for a in range(C.shape[0]):
                    for s in range(T.shape[1]):
                        indexes[i,a,s] = R[i,s] + gamma*V[i].dot(T[i,s,a])



        # SampleLam
        if policy_option == 27:
            
            pass

    


        blam_data = {}
        # BLam
        if policy_option in [30, 31, 32]:
            
            LAMBDA_EPSILON = 1e-6#output_data['epsilon_blam']

            start = time.time()

            indexes = np.zeros((N, C.shape[0], T.shape[1]))

            V_lam_0 = np.zeros((N, T.shape[1]))
            V_lam_eps = np.zeros((N, T.shape[1]))
            V_lam_final = np.zeros((N, T.shape[1]))
            
            Q = np.zeros((N, C.shape[0], T.shape[1]))

            # get V when lam = 0
            for i in range(N):
                # Go from S,A,S to A,S,S
                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] = R[i]

                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                # mdp.setVerbose()
                mdp.run()
                V_lam_0[i] = np.array(mdp.V)

            # get V when lam = epsilon
            for i in range(N):
                # Go from S,A,S to A,S,S
                T_i = np.swapaxes(T[i],0,1)
                R_i = np.zeros(T_i.shape)
                for x in range(R_i.shape[0]):
                    # subtract the lambda-weighted cost of acting
                    R_i[x] -= LAMBDA_EPSILON*C[x]
                    for y in range(R_i.shape[1]):
                        R_i[x,:,y] += R[i]


                # mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, initial_value=(V_lam_0[i]/2).tolist(), discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                # mdp.setVerbose()
                mdp.run()

                V_lam_eps[i] = np.array(mdp.V)


            blam_data['V_lam_0'] = V_lam_0
            blam_data['V_lam_eps'] = V_lam_eps
            blam_data['LAMBDA_EPSILON'] = LAMBDA_EPSILON


        # BLam
        if policy_option in [33, 34, 35, 36, 37, 38, 39, 40]:

            run_debug = False

            N_states = T.shape[1]

            start = time.time()

            epsilon_shift = 1e-3

            bound_grid_seed = np.array([0, 1e-1, 0.2])
            if output_data['blam_nonSL']:
                bound_grid_seed = np.array([0, 1e-1, 0.2, 0.5])
            # bound_grid_seed = np.linspace(0,1,10)
            # bound_grid_seed = np.array([0]+ [2**n for n in range(-4, -1)])
            # print(bound_grid_seed)
            # 1/0
            bound_grid = np.ones((N, len(bound_grid_seed)+1))*(-1)
            num_grid_points_per_process = np.zeros(N, dtype=int)

            # set the first grid points to be early in the slope
            bound_grid[:, :-1] = bound_grid_seed

            grid_lengths = np.zeros(bound_grid.shape)
            bound_grid_shift_left_once = np.zeros(bound_grid.shape)
            bound_grid_shift_left_once[:, :-2] = bound_grid_seed[1:]
            bound_grid_shift_left_once[:, -2] = bound_grid_seed[-1]*2


            debug_grid_seed = np.linspace(0,1,100)
            debug_grid = np.ones((N, len(debug_grid_seed)))*(-1)
            debug_grid[:, :] = debug_grid_seed



            n_grid_squares = bound_grid.shape[0]
            V_slope_grid_ub = np.zeros((N, n_grid_squares, N_states))
            V_intercept_grid_ub = np.zeros((N, n_grid_squares, N_states))

            V_slope_grid_lb = np.zeros((N, n_grid_squares, N_states))
            V_intercept_grid_lb = np.zeros((N, n_grid_squares, N_states))

            V_slope_grid_point_helper = np.zeros((N, n_grid_squares, N_states))
            V_slope_grid_point_plus_epsilon_helper = np.zeros((N, n_grid_squares, N_states))
            

            n_grid_squares_debug = debug_grid.shape[0]
            V_slope_grid_debug = np.zeros((N, n_grid_squares_debug, N_states))
            V_intercept_grid_debug = np.zeros((N, n_grid_squares_debug, N_states))
            V_slope_grid_point_helper_debug = np.zeros((N, n_grid_squares_debug, N_states))
            V_slope_grid_point_plus_epsilon_helper_debug = np.zeros((N, n_grid_squares_debug, N_states))

            # Go from S,A,S to A,S,S
            T_flipped = np.swapaxes(T, 1, 2)

            # Have to be careful with how we place the intercepts, y is a helper
            y_prev = np.zeros((N, N_states))
            for i in range(N):
                for lambda_ind in range(bound_grid_seed.shape[0]):
                    T_i = T_flipped[i]
                    R_i = np.zeros(T_i.shape)
                    lambda_val = bound_grid[i, lambda_ind]
                    for x in range(R_i.shape[0]):
                        # subtract the lambda-weighted cost of acting
                        R_i[x] -= lambda_val*C[x]
                        for y in range(R_i.shape[1]):
                            R_i[x,:,y] += R[i]

                    mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                    # mdp.setVerbose()
                    mdp.run()
                    V_slope_grid_point_helper[i, lambda_ind] = np.array(mdp.V)

                    if lambda_ind == 0:
                        y_prev[i] = V_slope_grid_point_helper[i, lambda_ind] 


                    R_i = np.zeros(T_i.shape)
                    lambda_val_plus_eps = bound_grid[i, lambda_ind] + epsilon_shift
                    for x in range(R_i.shape[0]):
                        # subtract the lambda-weighted cost of acting
                        R_i[x] -= lambda_val_plus_eps*C[x]
                        for y in range(R_i.shape[1]):
                            R_i[x,:,y] += R[i]

                    mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                    # mdp.setVerbose()
                    mdp.run()

                    V_slope_grid_point_plus_epsilon_helper[i, lambda_ind] = np.array(mdp.V)

                    V_slope_grid_ub[i, lambda_ind] = (V_slope_grid_point_plus_epsilon_helper[i, lambda_ind] -  V_slope_grid_point_helper[i, lambda_ind]) / epsilon_shift
                    # b = y - mx
                    V_intercept_grid_ub[i, lambda_ind] = y_prev[i] - V_slope_grid_ub[i, lambda_ind]*lambda_val
                    # y = mx + b
                    y_prev[i] = V_slope_grid_ub[i, lambda_ind]*bound_grid[i, lambda_ind+1] + V_intercept_grid_ub[i, lambda_ind]
                    
                    # if lambda_ind > 0:
                    #     V_slope_grid_lb[lambda_ind - 1, i] = (V_slope_grid_point_helper[lambda_ind, i] - V_slope_grid_point_helper[lambda_ind - 1, i]) / (bound_grid[lambda_ind, i] - bound_grid[lambda_ind - 1, i])
                    #     # b = y - mx
                    #     V_intercept_grid_lb[lambda_ind - 1, i] = (V_slope_grid_point_helper[lambda_ind - 1, i] - V_slope_grid_lb[lambda_ind - 1, i]*bound_grid[lambda_ind - 1, i])



                    num_grid_points_per_process[i] += 1
            ###
            #
            # Now check the processes that still have large slope and run sampeLam to get last grid point
            #
            ###
            # check the last slope of each process
            for proc, state_slopes in enumerate(V_slope_grid_ub[:, -2]):
                if (abs(state_slopes) > 1).any() and (not output_data['blam_nonSL']):


                    B_unit = B/N

                    V_vals, lambda_val, _ = lp_methods_gurobi.sample_lam(T[[proc]], R[[proc]], C, B_unit, start_state[[proc]].astype(int), gamma=gamma)
                    # lambda_val = 1
                    if lambda_val > bound_grid[-2, proc]:

                        bound_grid[proc, -1] = lambda_val
                        bound_grid_shift_left_once[proc, -2] = lambda_val
                        bound_grid_shift_left_once[proc, -1] = lambda_val*2

                        T_i = T_flipped[proc]
                        R_i = np.zeros(T_i.shape)

                        for x in range(R_i.shape[0]):
                            # subtract the lambda-weighted cost of acting
                            R_i[x] -= lambda_val*C[x]
                            for y in range(R_i.shape[1]):
                                R_i[x,:,y] += R[proc]

                        mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                        # mdp.setVerbose()
                        mdp.run()
                        
                        V_slope_grid_point_helper[proc, -1] = np.array(mdp.V)
                        # V_slope_grid_point_helper[-1, proc] = V_vals

                        R_i = np.zeros(T_i.shape)
                        lambda_val_plus_eps = lambda_val + epsilon_shift
                        for x in range(R_i.shape[0]):
                            # subtract the lambda-weighted cost of acting
                            R_i[x] -= lambda_val_plus_eps*C[x]
                            for y in range(R_i.shape[1]):
                                R_i[x,:,y] += R[proc]

                        mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                        # mdp.setVerbose()
                        mdp.run()
                        V_slope_grid_point_plus_epsilon_helper[proc, -1] = np.array(mdp.V)


                        lambda_ind = -1
                        V_slope_grid_ub[proc, lambda_ind] = (V_slope_grid_point_plus_epsilon_helper[proc, lambda_ind] -  V_slope_grid_point_helper[proc, lambda_ind]) / epsilon_shift


                        # update y_prev to reflect the new value of lambda we're using
                        y_prev[proc] = V_slope_grid_ub[proc, lambda_ind - 1]*bound_grid[proc, lambda_ind] + V_intercept_grid_ub[proc, lambda_ind - 1]

                        V_intercept_grid_ub[proc, lambda_ind] = (y_prev[proc] - V_slope_grid_ub[proc, lambda_ind]*lambda_val)


                        # # add last non-zero slope
                        # lambda_ind = -1
                        # V_slope_grid_lb[lambda_ind - 1, proc] = (V_slope_grid_point_helper[lambda_ind, proc] - V_slope_grid_point_helper[lambda_ind - 1, proc]) / (lambda_val - bound_grid[lambda_ind - 1, proc])
                        # # b = y - mx
                        # V_intercept_grid_lb[lambda_ind - 1, proc] = (V_slope_grid_point_helper[lambda_ind - 1, proc] - V_slope_grid_lb[lambda_ind - 1, proc]*bound_grid[lambda_ind - 1, proc])


                        num_grid_points_per_process[proc] += 1





            # Add the lower bounds, moving in reverse  (first one is easy, just a zero slope)              


            # print(V_intercept_grid_lb[0,lambda_ind])

            
            for proc in range(N):

                # computing the intercepts will be the secret sauce of forming the lower bounds
                # y values will need to be iteratively computed, but first one is 
                # on the actual V at the last grid point

                lambda_ind = num_grid_points_per_process[proc] - 1
                # print(lambda_ind)
                # add the zero slope and intercept
                V_slope_grid_lb[proc, lambda_ind] = 0
                V_intercept_grid_lb[proc, lambda_ind] = V_slope_grid_point_helper[proc, lambda_ind]


                y = V_slope_grid_lb[proc, lambda_ind]*bound_grid[proc, lambda_ind] + V_intercept_grid_lb[proc, lambda_ind]

                # Add the lower bounds, moving in reverse  
                for lambda_ind in range(0, num_grid_points_per_process[proc] - 1)[::-1]:
                    # print(lambda_ind)

                    # same slope as the ub one step above
                    V_slope_grid_lb[proc, lambda_ind] = V_slope_grid_ub[proc, lambda_ind+1]
                    # b = y - mx
                    V_intercept_grid_lb[proc, lambda_ind] = y - V_slope_grid_lb[proc, lambda_ind]*bound_grid[proc, lambda_ind+1]

                    # compute the next value of y to use for the next bound
                    # y = mx + b
                    y = V_slope_grid_lb[proc, lambda_ind]*bound_grid[proc, lambda_ind] + V_intercept_grid_lb[proc, lambda_ind]


            if run_debug:
                for lambda_ind in range(debug_grid.shape[1]):

                    for i in range(N):
                        T_i = T_flipped[i]
                        R_i = np.zeros(T_i.shape)
                        lambda_val = debug_grid[i, lambda_ind]
                        for x in range(R_i.shape[0]):
                            # subtract the lambda-weighted cost of acting
                            R_i[x] -= lambda_val*C[x]
                            for y in range(R_i.shape[1]):
                                R_i[x,:,y] += R[i]

                        mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                        # mdp.setVerbose()
                        mdp.run()
                        V_slope_grid_point_helper_debug[i, lambda_ind] = np.array(mdp.V)

                        R_i = np.zeros(T_i.shape)
                        lambda_val_plus_eps = debug_grid[i, lambda_ind] + epsilon_shift
                        for x in range(R_i.shape[0]):
                            # subtract the lambda-weighted cost of acting
                            R_i[x] -= lambda_val_plus_eps*C[x]
                            for y in range(R_i.shape[1]):
                                R_i[x,:,y] += R[i]

                        mdp = mdptoolbox.mdp.ValueIteration(T_i, R_i, discount=gamma, stop_criterion=output_data['mdp_stop_criterion'], epsilon=output_data['mdp_epsilon'])
                        # mdp.setVerbose()
                        mdp.run()

                        V_slope_grid_point_plus_epsilon_helper_debug[i, lambda_ind] = np.array(mdp.V)

                        V_slope_grid_debug[i, lambda_ind] = (V_slope_grid_point_plus_epsilon_helper_debug[i, lambda_ind] -  V_slope_grid_point_helper_debug[i, lambda_ind]) / epsilon_shift
                        # b = y - mx
                        V_intercept_grid_debug[i, lambda_ind] = (V_slope_grid_point_helper_debug[i, lambda_ind] - V_slope_grid_debug[i, lambda_ind]*lambda_val)
                        
                 
                plot_bounds(V_slope_grid_ub, V_intercept_grid_ub, V_slope_grid_lb,
                    V_intercept_grid_lb, num_grid_points_per_process, bound_grid,
                    V_slope_grid_debug=V_slope_grid_debug, V_intercept_grid_debug=V_intercept_grid_debug,
                    debug_grid=debug_grid)



            grid_lengths = bound_grid_shift_left_once - bound_grid


            blam_data['V_slope_grid_ub'] = V_slope_grid_ub
            blam_data['V_intercept_grid_ub'] = V_intercept_grid_ub
            
            blam_data['V_slope_grid_lb'] = V_slope_grid_lb
            blam_data['V_intercept_grid_lb'] = V_intercept_grid_lb

            blam_data['num_grid_points_per_process'] = num_grid_points_per_process
            blam_data['bound_grid'] = bound_grid
            blam_data['epsilon_shift'] = epsilon_shift
            blam_data['grid_lengths'] = grid_lengths





        # print("multi-index lp time:",time.time() - start)


    # Mean-Field
    if policy_option in [50, 51, 52, 53, 54]:
        if 'mf_clusters' not in output_data.keys(): output_data['mf_clusters'] = dict()
        output_data['mf_clusters'][policy_option] = None


    if start_state is not None:
        state_log[:,0] = start_state
    else:
        state_log[:,0] = 1


    #######  Run simulation #######
    print('Running simulation w/ policy: %s'%policy_option)
    # make array of nan to initialize observations
    observations = np.full(N, np.nan)
    # print("Policy:", policy_option)

    valid_action_combinations = None
    if policy_option in [2, 4]:
        options = np.array(list(product(np.arange(C.shape[0]), repeat=N)))
        valid_action_combinations = utils.list_valid_action_combinations(N,C,B,options)


    # for t in range(1,L):
    for t in tqdm.tqdm(range(1,L)):
        
        '''
        Learning T_hat from simulation so far
        '''
        #print("Round: %s"%t)
        days_remaining = L-t
        
        actions=getActions(N, T, R, C, B, k, valid_action_combinations=valid_action_combinations, current_state=state_log[:,t-1], 
                            optimal_policy=optimal_policy, gamma=gamma,
                             policy_option=policy_option, days_remaining=days_remaining, combined_state_dict=combined_state_dict,
                             indexes=indexes, output_data=output_data, blam_data=blam_data, 
                             mean_field_clusters=mean_field_clusters, simulation_length=simulation_length)
            
                
        actions_record[:, t-1]=actions    
        if action_logs is not None:
            action_logs[policy_option].append(actions.astype(int))


        #TODO: Modify T_hat to estimated value of T_hat (sliding window, etc.)

        state_log[:,t] = takeAction(state_log[:,t-1].reshape(-1), T, actions, random_stream=world_random_stream)



    return state_log



def worker(i, runtimes_mp, reward_logs_mp, 
        first_seedbase, first_world_seedbase, args, trial_percent_lam0, output_data,
        N, L, k, policies, size_limits, file_root, savestring):
    # use np global seed for rolling random data, then for random algorithmic choices
    seedbase = first_seedbase + i
    np.random.seed(seed=seedbase)

    # Use world seed only for evolving the world (If two algs 
    # make the same choices, should create the same world for same seed)
    world_seed_base = first_world_seedbase + i


    # print ("Seed is", seedbase)

    T = None
    R = None
    C = None
    B = None
    start_state = None

    if args.data =='strict_fast':
        REWARD_BOUND = 1
        T, R, C, B = simulation_environments.get_strict_fast_experiment(N, args.num_states, args.num_actions, REWARD_BOUND)
    if args.data =='strict_random':
        REWARD_BOUND = 1
        T, R, C, B = simulation_environments.get_strict_random_experiment(N, args.num_states, args.num_actions, REWARD_BOUND)
    if args.data =='puterman_random':
        REWARD_BOUND = 1
        T, R, C, B = simulation_environments.get_puterman_random_experiment(N, args.num_states, args.num_actions, REWARD_BOUND)

    if args.data =='full_random':
        REWARD_BOUND = 1
        start_state = np.zeros(N)
        T, R, C, B = simulation_environments.get_full_random_experiment(N, args.num_states, args.num_actions, REWARD_BOUND)

    if args.data =='eng1':
        REWARD_BOUND = 1
        B = args.budget_frac*N
        T, R, C, B = simulation_environments.get_eng1_experiment(N, args.num_states, args.num_actions, B, REWARD_BOUND)

    if args.data =='eng11':
        REWARD_BOUND = 1

        # percent_lam0 = trial_percent_lam0
        percent_lam0 = 0.5
        percent_greedy = (1 - percent_lam0) / 2

        B=round(N*(1-percent_greedy-percent_lam0))
        B=args.budget_frac*N
        T, R, C = simulation_environments.get_eng11_experiment(N, args.num_actions, percent_greedy, percent_lam0, REWARD_BOUND)
        args.num_states = T.shape[1]
        start_state = np.zeros(N)

    if args.data == 'adverse_lagrangian':
        T, R, C, B, start_state = simulation_environments.get_adverse_for_lagrangian(N)

    if args.data == 'adverse_mf':
        T, R, C, B, start_state = simulation_environments.get_adverse_for_mean_field(N)


    if args.data =='healthcare':

        B = N*args.budget_frac
        REWARD_BOUND = 1
        num_adherence_states = args.num_states
        IP_length = 2*args.num_states
        percent_good_patients = 0.64
        percent_unresponsive_patients = 0.01
        percent_responsive_patients = 0.175

        T, R, C, start_state = simulation_environments.get_healthcare_experiment(N, num_adherence_states, IP_length, percent_good_patients, 
                                        percent_unresponsive_patients, percent_responsive_patients, B, REWARD_BOUND)
        

    if args.data =='sl_sa':

        B = N*args.budget_frac
        REWARD_BOUND = 1
        percent_random = trial_percent_lam0
        # percent_random = 0.05

        num_adherence_states = args.num_states
        IP_length = 2*args.num_states
        percent_good_patients = 0.64
        percent_unresponsive_patients = 0.01
        percent_responsive_patients = 0.175


        S = num_adherence_states * (IP_length+2)
        A = 5
        T = np.zeros((N,S,A,S))
        R = np.zeros((N,S))
        start_state = np.ones(N)
        
        num_rand = int(N*percent_random)
        T[:num_rand], R[:num_rand], C, _ = simulation_environments.get_full_random_experiment(num_rand, S, A, REWARD_BOUND)
        
        num_healthcare = N - num_rand
        T[num_rand:], R[num_rand:], C, ss = simulation_environments.get_healthcare_experiment(num_healthcare, num_adherence_states, IP_length, percent_good_patients, 
                                        percent_unresponsive_patients, percent_responsive_patients, B, REWARD_BOUND)
        if len(ss) > 0:
            start_state[:] = ss[0]
        else:
            start_state[:] = num_adherence_states-1



    np.random.seed(seed=seedbase)

    runtimes = np.zeros(len(policies))
    action_logs = dict()
    for p, policy_option in enumerate(policies):

        # if policy_option == 30:
        #     output_data['epsilon_blam'] = args.epsilon_blam
        # if policy_option == 31:
        #     output_data['epsilon_blam'] = 0.5
        # if policy_option == 32:
        #     output_data['epsilon_blam'] = 1.0

        if policy_option == 33:
            output_data['epsilon_blam'] = 0.1
        if policy_option == 34:
            output_data['epsilon_blam'] = 0.2
        if policy_option == 35:
            output_data['epsilon_blam'] = 0.3
        if policy_option == 36:
            output_data['epsilon_blam'] = 0.5

        if policy_option == 37:
            output_data['epsilon_blam'] = 0.1
            output_data['blam_nonSL'] = True
        if policy_option == 38:
            output_data['epsilon_blam'] = 0.2
            output_data['blam_nonSL'] = True
        if policy_option == 39:
            output_data['epsilon_blam'] = 0.3
            output_data['blam_nonSL'] = True
        if policy_option == 40:
            output_data['epsilon_blam'] = 0.5
            output_data['blam_nonSL'] = True


        policy_start_time=time.time()
        if size_limits[policy_option]==None or size_limits[policy_option]>N:
            np.random.seed(seed=seedbase)

            optimal_policy = None
            combined_state_dict = None


            state_matrix=simulateAdherence(N, L, T, R, C, B, k, policy_option=policy_option, seedbase=seedbase, action_logs=action_logs,
                                               world_random_seed=world_seed_base, optimal_policy = optimal_policy, combined_state_dict=combined_state_dict,
                                               file_root=file_root, output_data=output_data, start_state=start_state, gamma=args.discount_factor,
                                               mean_field_clusters=args.mean_field_clusters, simulation_length=args.simulation_length)

            np.save(file_root+'/logs/adherence_log/adherence_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states, trial_percent_lam0), state_matrix)
            

            reward_matrix = np.zeros(state_matrix.shape)
            for ind_i in range(state_matrix.shape[0]):
                for ind_j in range(state_matrix.shape[1]):
                    reward_matrix[ind_i,ind_j] = (args.discount_factor**ind_j)*R[ind_i, state_matrix[ind_i, ind_j]]


            np.save(file_root+'/logs/adherence_log/rewards_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states,trial_percent_lam0), reward_matrix)

            reward_logs_mp[policy_option].append(np.sum(reward_matrix))

            np.set_printoptions(precision=None, threshold=np.inf, edgeitems=None, 
                linewidth=np.inf, suppress=None, nanstr=None, 
                infstr=None, formatter=None, sign=None, 
                floatmode=None, legacy=None)


        policy_end_time=time.time()
        policy_run_time=policy_end_time-policy_start_time
        np.save(file_root+'/logs/runtime/runtime_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states,trial_percent_lam0), policy_run_time)

        runtimes[p] = policy_run_time

    ##### SAVE ALL RELEVANT LOGS #####
    runtimes_mp.append(runtimes)

    # write out action logs
    for policy_option in action_logs.keys():
        fname = os.path.join(args.file_root,'logs/action_logs/action_logs_%s_N%s_b%s_L%s_policy%s_data%s_seed%s_S%s_pl0%s.csv'%(savestring, N,args.budget_frac,L,policy_option,args.data,seedbase,args.num_states,trial_percent_lam0))
        columns = list(map(str, np.arange(N)))
        df = pd.DataFrame(action_logs[policy_option], columns=columns)
        df.to_csv(fname, index=False)


if __name__=="__main__":

    parser = argparse.ArgumentParser(description='Run adherence simulations with various methods.')
    parser.add_argument('-n', '--num_patients', default=2, type=int, help='Number of Processes')
    parser.add_argument('-b', '--budget_frac', default=0.5, type=float, help='Budget per day as fraction of n')
    parser.add_argument('-l', '--simulation_length', default=180, type=int, help='Number of days to run simulation')
    parser.add_argument('-N', '--num_trials', default=5, type=int, help='Number of trials to run')
    parser.add_argument('-S', '--num_states', default=2, type=int, help='Number of states per process')
    parser.add_argument('-A', '--num_actions', default=2, type=int, help='Number of actions per process')
    parser.add_argument('-g', '--discount_factor', default=0.95, type=float, help='Discount factor for MDP solvers')
    parser.add_argument('-mfc', '--mean_field_clusters', default=0, type=int, help='Number of clusters for mean-field. Default = 0 => no clustering.')

    parser.add_argument('-d', '--data', default='real', choices=['strict_fast','strict_random',
        'puterman_random','full_random','eng1','eng11','healthcare','sl_sa', 'adverse_lagrangian', 'adverse_mf'], 
        type=str,help='Method for generating transition probabilities')

    parser.add_argument('-ip', '--ip_length', default=2, type=int, help='Length of Intensive Phase for healthcare experiment')

    parser.add_argument('-msc', '--mdp_stop_criterion', default='full', choices=['full','fast'], type=str, help='Which stop criterion to use for value iteration')
    parser.add_argument('-me', '--mdp_epsilon', default=1e-1, type=float, help='Tolerance for Value Iteration')

    parser.add_argument('-eb', '--epsilon_blam', default=0.2, type=float, help='Bound tightness for BLam')

    parser.add_argument('-s', '--seed_base', type=int, help='Base for the random seed')
    parser.add_argument('-ws','--world_seed_base', default=None, type=int, help='Base for the random seed')

    parser.add_argument('-f', '--file_root', default='./..', type=str,help='Root dir for experiment (should be the dir containing this script)')
    parser.add_argument('-pc', '--policy', default=-1, type=int, help='policy to run, default is all policies')
    parser.add_argument('-tr', '--trial_number', default=None, type=int, help='Trial number')
    parser.add_argument('-sv', '--save_string', default='', type=str, help='special string to include in saved file name')

    parser.add_argument('-sid', '--slurm_array_id', default=None, type=int, help='Specify the index of the parameter combo')

    args = parser.parse_args()

    
    ##### File root
    if args.file_root == '.':
        args.file_root = os.getcwd()

    ##### Save special name
    if args.save_string == '':
        args.save_string = str(time.ctime().replace(' ', '_').replace(':','_'))

    policies_to_plot = None

    ##### Policies to run
    if args.policy < 0:
        #************** 
        policies = [0, 51, 52, 53, 54, 37, 38, 24, 27, 21]
        policies_to_plot = policies


    else:
        policies=[args.policy]

    N = 0
    k = 0


    NUM_TRIALS = 20
    trial_number_list = [i for i in range(NUM_TRIALS)]

    n_list = [250, 500, 750]
    budget_frac_list = [0.1, 0.2, 0.5]
    state_size_list = [3, 4, 5]



    # master_combo_list = list(itertools.product(trial_number_list, n_list, percent_lam0_list, n_actions_list))
    # master_combo_list = list(itertools.product(trial_number_list, n_list, budget_list, n_states_list, percent_random_list))
    master_combo_list = list(itertools.product(trial_number_list, n_list, budget_frac_list, state_size_list))

    # print(master_combo_list[args.slurm_array_id]);1/0
    # print(len(master_combo_list));1/0

    trial_percent_lam0 = 0.25
    if args.slurm_array_id is not None:
        combo = master_combo_list[args.slurm_array_id]

        args.trial_number = combo[0]
        args.num_patients = combo[1] # num processes
        args.budget_frac = combo[2]
        args.num_states = combo[3]
        # trial_percent_lam0 = combo[4]




    # If we pass a trial number, that means we are running this as a job
    # and we want jobs/trials to run in parallel so this does some rigging to enable that,
    # while still synchronizing all the seeds
    if args.trial_number is not None:
        args.num_trials=1
        add_to_seed_for_specific_trial=args.trial_number
    else:
        add_to_seed_for_specific_trial=0

    first_seedbase=np.random.randint(0, high=100000)
    if args.seed_base is not None:
        first_seedbase = args.seed_base+add_to_seed_for_specific_trial

    first_world_seedbase=np.random.randint(0, high=100000)
    if args.world_seed_base is not None:
        first_world_seedbase = args.world_seed_base+add_to_seed_for_specific_trial


    N=args.num_patients
    L=args.simulation_length
    k=0
    savestring=args.save_string
    N_TRIALS=args.num_trials
    
    
    # record_policy_actions=[2, 3, 4, 19, 20, 21, 22, 23, 24, 25]
    record_policy_actions=[5, 6, 7, 21, 27, 24, 30, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 50, 51, 52, 53, 54]

    size_limits={   
                    0:None, 2:8, 3:None, 4:8, 
                    5:None, 6:None, 7:None,
                    19:None, 20:None, 21:None, 22:None, 23:None,
                    24:None, 25:None, 26:None, 27:None, 28:None,
                    30:None, 31:None, 32:None, 33:None, 34:None,
                    35:None, 36:None, 37:None, 38:None, 39:None,
                    40:None,
                    50:None, 51:None, 52:None, 53:None, 54:None,
                }

    # policy names dict
    pname={
            0: 'nobody',    2: 'Random',
            5: 'RoundRobin1', 6:'RoundRobin2',
            7: 'RoundRobinE',
            21:'Hawkins',

            24:'VfNc',


            27:'SampleLam',

            # 30:'Blam0.2',
            # 31:'Blam0.5',
            # 32:'Blam1.0',

            33:'BLamSL0.1',
            34:'BLamSL0.2',
            35:'BLamSL0.3',
            36:'BLamSL0.5',

            37:'BLam0.1',
            38:'BLam0.2',
            39:'BLam0.3',
            40:'BLam0.5',

            50:f'MFclus{args.mean_field_clusters}',
            51:'MFnoClus',
            52:'MFclus20',
            53:'MFclus50',
            54:'MFclus100'
        }



    # for rapid prototyping
    # use this to avoid updating all the function calls when you need to pass in new
    # algo-specific things or return new data
    output_data = {}

    # list because one for each trial
    output_data['hawkins_lambda'] = []
    output_data['decoupled_lambdas'] = []
    output_data['iterative_lambdas'] = []
    output_data['only_goods_lambda'] = []

    output_data['good_percent'] = 0.01

    output_data['lambdas_samplelam'] = []
    output_data['nsamples_samplelam'] = np.log2(N)
    # output_data['nsamples_samplelam'] = N/4
    # output_data['nsamples_samplelam'] = N


    output_data['mdp_stop_criterion'] = args.mdp_stop_criterion
    output_data['mdp_epsilon'] = args.mdp_epsilon


    output_data['blam_ubs'] = []
    output_data['blam_lbs'] = []
    output_data['blam_xvals'] = []
    output_data['blam_lambda_out'] = []
    output_data['epsilon_blam'] = args.epsilon_blam
    output_data['blam_nonSL'] = False



    
    manager = multiprocessing.Manager()
    
    runtimes_mp = manager.list()
    reward_logs_mp = manager.dict([(key,manager.list()) for key in pname.keys()])

    start=time.time()
    file_root=args.file_root

    jobs = []
    for i in range(N_TRIALS):
        proc = multiprocessing.Process(target=worker, args=(i, runtimes_mp, reward_logs_mp, 
                                    first_seedbase, first_world_seedbase, args, trial_percent_lam0, output_data, 
                                    N, L, k, policies, size_limits, file_root, savestring))
        jobs.append(proc)
        # proc.start()  # uncomment this for parallel, and comment the one below

    for proc in jobs:
        proc.start()  # uncomment this for sequential, and comment the one above
        proc.join()

    runtimes = np.array(runtimes_mp)
    reward_logs = dict(reward_logs_mp)

    end=time.time()
    print ("Time taken: ", end-start)
    
    for i,p in enumerate(policies):
        # print (pname[p],": ", np.mean(state_log[p]))
        print (pname[p],": ", runtimes[:,i].mean())

    # exit()


    if args.policy<0:


        bottom = 0
        labels = [pname[i] for i in policies_to_plot]
        values=[round(np.mean(np.array(reward_logs[i])/N)-bottom, 4) for i in policies_to_plot]
        values = np.array(values)
        values -= values[0]  # Abheek: remove this for real scale plots
        errors=[np.std(np.array(reward_logs[i])/N) for i in policies_to_plot]

        vals = [values, errors]
        df = pd.DataFrame(vals, columns=labels)
        fname = os.path.join(args.file_root,'logs/results/results_%s_N%s_b%s_L%s_data%s_S%s_pl0%s.csv'%(savestring, N,args.budget_frac,L,args.data,args.num_states,trial_percent_lam0))
        df.to_csv(fname,index=False)

        
        utils.barPlot(labels, values, errors, ylabel='Discounted sum of rewards',
            title='%s arms, %s budget per day; trials: %s' % (N, N*args.budget_frac, N_TRIALS),
            filename='img/results_%s_N%s_b%s_L%s_data%s_S%s_pl0%s.png'%(savestring, N,args.budget_frac,L,args.data,args.num_states,trial_percent_lam0), root=args.file_root,
            bottom=0)


        # utils.plotLambdas(output_data['hawkins_lambda'], output_data['lambdas_samplelam'],
        #     filename='bad_decoupled_lambda.png',
        #     root=args.file_root)

        # utils.plotIterativeLambdas(output_data['hawkins_lambda'], output_data['iterative_lambdas'],
        #     filename='img/iterative_lambdas_'+savestring+'_N%s_k%s_trials%s_data%s_s%s_lr%s_%s.png'%(N,k,N_TRIALS, args.data,first_seedbase, LEARNING_MODE, '%s'), 
        #     root=args.file_root, only_goods_lambda=output_data['only_goods_lambda'])
        
        # utils.plotIterativeLambdas(output_data['hawkins_lambda'], output_data['lambdas_qlam'],
        #     filename='img/iterative_lambdas_'+savestring+'_N%s_k%s_trials%s_data%s_s%s_lr%s_%s.png'%(N,k,N_TRIALS, args.data,first_seedbase, LEARNING_MODE, '%s'), 
        #     root=args.file_root)

        # for l,u in zip(output_data['blam_lbs'],output_data['blam_ubs']):
        #     print(l)
        #     print(u)
        #     print()

        # print(output_data['blam_lambda_out'])
        # print(output_data['hawkins_lambda'])

        # utils.plotBLambdas(output_data['blam_lbs'], output_data['blam_ubs'], output_data['blam_xvals'], output_data['hawkins_lambda'],
        #     filename='img/blambdas_'+savestring+'_N%s_k%s_trials%s_data%s_s%s_lr%s_%s.png'%(N,k,N_TRIALS, args.data,first_seedbase, LEARNING_MODE, '%s'), 
        #     root=args.file_root)
        


