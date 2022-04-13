from ode import *

def main():
	""" """
	# settings for V-Rho and B-X plots:
	Settings.GRAPH_COLOUR = 'red'
	Settings.EPSILON = 0.2

	solveForSingleEpsilon()

	# settings for R-Epsilon and B-Epsilon plots:
	Settings.NUM_RHOS = 10000
	Settings.NUM_EPSILONS = 10

	Settings.ENABLE_ANALYTIC_PLOT = True
	Settings.ENABLE_NUMERICAL_PLOT = True
	
	solveForEpsilonArray()

if __name__ == "__main__":
	""" """
	plt.figure(figsize=(0.5*16, 0.5*10))
	main()