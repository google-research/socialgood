import utils
import matplotlib.pyplot as plt
import csv
import numpy as np



for narms in [50]:
	with open(f"a10.csv") as fd:
	    rd = csv.reader(fd, delimiter=",")
	    rd = list(rd)
	    # for row in rd:
	    #     print(row)

	# print(rd)

	labels = rd[0][1:]
	values = np.array([float(x) for x in rd[1][1:]])
	errors = np.array([float(x) for x in rd[2][1:]])
	x_pos = np.arange(len(labels))

	# print(labels)
	# print(values)
	# print(errors)

	# Build the plot
	fig, ax = plt.subplots()
	ax.bar(x_pos, values, yerr=errors, alpha=0.5, ecolor='black', capsize=10)
	ax.set_ylabel("Discounted Sum of Rewards (minus Nobody's Reward)")
	ax.set_xticks(x_pos)
	ax.set_xticklabels(labels)
	# ax.set_title('Coefficent of Thermal Expansion (CTE) of Three Metals')
	# ax.yaxis.grid(True)

	# Save the figure and show
	plt.tight_layout()
	# plt.savefig('bar_plot_with_error_bars.png')
	plt.show()

	# utils.barPlot(labels, values, errors, ylabel='Discounted Sum of Rewards', bottom=0)



