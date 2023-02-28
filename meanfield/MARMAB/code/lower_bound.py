import math
import numpy as np
import scipy.special as sp
import matplotlib.pyplot as plt



def nchoosek(m, k):
	global Binom

	if Binom[m, k] < 0:
		Binom[m, k] = math.comb(m, k)

	return Binom[m, k]


def f(t, m):
	global F

	if t < 1 or t > T:
		assert False, ""

	if F[t,m] >= 0:
		return F[t,m]

	# print(t, m)

	A = np.zeros(m)
	for i in range(m):  # i players end up in reward 1 state
		A[i] = min(n, i)  # reward for this timestep
		if t < T:
			A[i] += f(t + 1, m - i + min(n, i))  # reward for future
		A[i] *= nchoosek(m, i) / 2**m  # multiply by probability

	F[t,m] = A.sum()
	return F[t,m]


def main():
	global n, T, F, Binom

	n = 1000
	Binom = -np.ones((2*n+1, 2*n+1))

	# T = 100
	# F = -np.ones((T+1, 2*n+1))

	# plot T vs error (hopefully T^2 dependency)
	T_array = np.arange(1,101)
	E_array = np.zeros(T_array.shape)
	for T in T_array:
		print(T)
		F = -np.ones((T+1, 2*n+1))
		E_array[T-1] = f(1, 2*n)

	E_array = T_array*n - E_array

	np.save('EvsT.npy', E_array)
	plt.plot(T_array, E_array)
	plt.show()


def plot_only():
	pass
	# E_array = np.load('EvsT.npy')
	# T_array = np.arange(1,101)
	# E_array = T_array*100 - E_array
	
	# plt.plot(T_array, E_array)
	# plt.show()


def alternate_dp():
	n = 1000
	T = 100

	# preprocessing, compute the binom coeff / prob
	prob = np.zeros((2*n+1, 2*n+1))
	for i in range(2*n+1):
		for j in range(i+1):
			prob[i,j] = sp.comb(i, j, exact=True) / 2**i
		print(f"done preprocessing {i}")

	# DP
	F = np.zeros((T+1, 2*n+1))
	for t in range(1, T+1):
		for m in range(2*n+1):
			i_values = np.arange(m+1)
			F[t,m] = (prob[m, i_values] * 
				(np.minimum(n, i_values) + F[t-1, m - i_values + np.minimum(n, i_values)]) ).sum()
		print(f"done t = {t}")

	t_values = np.arange(T+1)
	errors = n*t_values - F[:,2*n]

	np.save('EvsT.npy', errors)
	plt.plot(t_values, errors)
	plt.show()



if __name__ == '__main__':
	# main()
	# plot_only()
	alternate_dp()
	
