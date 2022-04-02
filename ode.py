"""

All mathematical variables used are the "tilde-variables", e.g., phi-tilde not
phi.

The user only needs to adjust main() (cf. class::Settings)

"""

import os
import numpy as np
import matplotlib.pylab as plt
from math import pi
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.signal import argrelextrema as findExtrema


# renaming the function 'fsolve' to 'getZero' for clarity.
# Takes a function and an x coordinate, returns x0 closest to x such that f(x0) = 0.
def getZero(of, around): return fsolve(of, around)[0];

def V(phi):
	""" Returns an array of V = V(phi, epsilon) in the range of phi. """
	global Settings

	return (1/8)*(phi**2 - 1)**2 + (Settings.EPSILON/2)*(phi - 1)

def dV_dPhi(phi):
	""" Returns dV/dPhi in the range of phi. """
	global Settings

	return (phi**3 - phi + Settings.EPSILON) / 2

def ddPhi(initialConditions, rho, potentialShift, dummy):
	""" Returns phi'' in the range of rho. """
	global Settings

	phi = initialConditions[0]
	dPhi = initialConditions[1]
	ddPhi = -(3/rho) * dPhi + dV_dPhi(phi)
	dB = 2*(pi**2) * (rho**3) * ((1/2)*(dPhi**2) + (1/8)*(phi**2 - 1)**2 + (Settings.EPSILON/2)*(phi - 1) + potentialShift)
	
	return dPhi, ddPhi, dB


def getConvergingPhi(rho):
	""" Determine initial phi that makes phi converge to a local maximum
		in potential V using binary search and the shooting method """

	global Settings

	def isPhiDivergent():
		phi = solution[:, 0]

		return max(phi) > 2 or min(phi) < -2

	def isPhiOscillatory():
		phi = solution[:, 0]

		for i in range(len(phi)-1):
			if phi[i] > phi[i+1]:
				return True

		return False

	# set lower & upper bounds of phi0
	minPhi0 = getZero(of=dV_dPhi, around=-1)
	maxPhi0 = getZero(of=dV_dPhi, around=0)
	middlePhi0 = (minPhi0 + maxPhi0) / 2

	# calculate potential shift to make V(phi-) = 0
	falseVacuumIndex = findExtrema(V(rho), np.less)[0][0]
	potentialShift = -1 * V(rho)[falseVacuumIndex]

	if Settings.ENABLE_DEBUGGING:
		print(findExtrema(V(rho), np.less))

	# odeint returns [[phi0, dPhi0, B0], [phi1, dPhi1, B1], ...]
	solution = odeint(ddPhi, [middlePhi0, 0, 0], rho, args=(potentialShift, 0))

	# binary search for finding phi0
	for i in range(100):

		if isPhiDivergent():
			minPhi0 = middlePhi0

		elif isPhiOscillatory():
			maxPhi0 = middlePhi0

		middlePhi0 = (minPhi0 + maxPhi0) / 2
		solution = odeint(ddPhi, [middlePhi0, 0, 0], rho, args=(potentialShift, 0))

	# return phi0, phi, dPhi and B
	return middlePhi0, solution[:, 0], solution[:, 1], solution[:, 2]

def getConvergingB(rho, B):
	""" Finds and returns a value B converges to by comparing local sum-squared-
		errors. Based on the assumption that the true converging point would have
		the smallest SSE around the point.

		The name of this function is misleading since this function 'getConvergingB'
		does something different to 'getConvergingPhi'. """

	global Settings

	windowLength = int(len(rho) / 10)
	arrSSE = [] # array to save sum squared errors

	# slide the window along rho and calculate SSE of B in each window
	for i in range(len(B) - windowLength + 1):
		localValues = B[i : i+windowLength]
		avg = sum(localValues) / windowLength

		sse = np.sum((localValues[:] - avg) ** 2)
		arrSSE.append(sse)

	# find the index of lowest SSE and return B at that point
	return B[np.argmin(arrSSE) + int(windowLength/2)]


def plotAndSaveV_Phi():
	global Settings

	x = np.linspace(-1.5, 1.5, 1000)
	y = -V(x)

	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axvline(x=0, color='black', linewidth=0.5)
	plt.axvline(x=getZero(of=dV_dPhi, around=-1), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.axvline(x=getZero(of=dV_dPhi, around=0), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.axvline(x=getZero(of=dV_dPhi, around=1), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.plot(x, y, color=Settings.GRAPH_COLOUR, linewidth=Settings.LINE_WIDTH,)

	plt.axis([min(x), max(x), min(y), 1.2*max(y)])
	plt.xlabel(r'$\~{\phi}$', fontsize=15)
	plt.ylabel(r'$-\~V$', fontsize=15)

	fileName = 'v_phi_' + str(Settings.EPSILON) + '.png'
	plt.savefig(fileName, format='png', dpi=350)

def plotAndSavePhi_Rho(x, y):
	global Settings

	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=getZero(of=dV_dPhi, around=1), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.axhline(y=getZero(of=dV_dPhi, around=0), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.axhline(y=getZero(of=dV_dPhi, around=-1), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.plot(x, y, color=Settings.GRAPH_COLOUR, linewidth=Settings.LINE_WIDTH,)

	plt.axis([0, max(x), -1.5, 1.5])
	plt.xlabel(r'$\~{\rho}$', fontsize=15)
	plt.ylabel(r'$\~{\phi}$', fontsize=15)

	fileName = 'phi_rho_' + str(Settings.EPSILON) + '.png'
	plt.savefig(fileName, format='png', dpi=350)

def plotAndSaveR_Epsilon(x, y):
	global Settings

	plt.clf()

	if Settings.ENABLE_NUMERICAL_PLOT:
		plt.plot(x, y, color=Settings.GRAPH_COLOUR, linestyle='', marker=Settings.MARKER_STYLE, markersize=Settings.MARKER_SIZE)

	if Settings.ENABLE_ANALYTIC_PLOT:
		lower, upper = Settings.ANALYTIC_R_EPSILON_RANGE
		x = np.linspace(lower, upper, 200)
		y = 2 / x
		plt.plot(x, y, color=Settings.GRAPH_COLOUR, linewidth=Settings.LINE_WIDTH, linestyle='-')

	plt.xticks([0.1*n for n in range(10)])
	plt.xlabel(r'$\~{\epsilon}$', fontsize=15)
	plt.ylabel(r'$R$', fontsize=15)
	plt.axis([0, 0.42, 0, 1.1*max(y)])

	fileName = 'r_epsilon_' + str(Settings.NUM_EPSILONS) + '.png'
	plt.savefig(fileName, format='png', dpi=350)

def plotAndSaveB_X(x, y):
	global Settings

	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=getConvergingB(x, y), color='grey', linewidth=Settings.DOTTED_LINE_WIDTH, linestyle=Settings.DOTTED_LINE_STYLE)
	plt.plot(x, y, color=Settings.GRAPH_COLOUR, linewidth=Settings.LINE_WIDTH, linestyle='-')

	plt.xlabel(r'$x$', fontsize=15)
	plt.ylabel(r'$B$', fontsize=15)
	plt.axis([0, max(x), 1.2*min(y), 1.5*getConvergingB(x, y)])

	fileName = 'b_x_' + "{:.2f}".format(Settings.EPSILON) + '.png'
	plt.savefig(fileName, format='png', dpi=350)
	
def plotAndSaveB_Epsilon(x, y):
	global Settings

	plt.clf()

	if Settings.ENABLE_NUMERICAL_PLOT:
		plt.axhline(y=0, color='black', linewidth=0.5)
		plt.plot(x, y, color=Settings.GRAPH_COLOUR, linestyle='', marker=Settings.MARKER_STYLE, markersize=Settings.MARKER_SIZE)

	if Settings.ENABLE_ANALYTIC_PLOT:
		lower, upper = Settings.ANALYTIC_B_EPSILON_RANGE
		x = np.linspace(lower, upper, 200)
		y = 27 * (pi**2) * (2/3)**4 / (2 * x**3)
		plt.plot(x, y, color=Settings.GRAPH_COLOUR, linewidth=Settings.LINE_WIDTH, linestyle='-')

	plt.xticks([0.1*n for n in range(10)])
	plt.xlabel(r'$\~{\epsilon}$', fontsize=15)
	plt.ylabel(r'$\~{B}$', fontsize=15)
	plt.axis([0, 0.44, -0.1*max(y), 1.1*max(y)])

	fileName = 'b_epsilon_' + str(Settings.NUM_EPSILONS) + '.png'
	plt.savefig(fileName, format='png', dpi=350)


def solveForSingleEpsilon():
	""" Solves the bubble equation to get phi0 and phi for a given epsilon. Also
		generates and saves phi-rho and V-phi plots. """

	global Settings

	# initialise varibles
	rho = np.linspace(1e-9, 50, Settings.NUM_RHOS)

	# solve ODE
	phi0, phi, dPhi, B = getConvergingPhi(rho)

	# save results
	plotAndSaveV_Phi()
	plotAndSavePhi_Rho(rho, phi)
	plotAndSaveB_X(rho, B)

	print("[ode.py] Phi_initial = {:f} for epsilon = {:f}".format(phi0, Settings.EPSILON))
	print("[ode.py] (V-phi, phi-rho and B-x saved in {0:s})".format(os.getcwd()))

def solveForEpsilonArray():
	""" Solves the bubble equation for a given range of epsilon. Also generates
		and saves an R-versus-epsilon plot. """

	global Settings

	# initialise variables
	arrR = []
	arrB = []
	arrEpsilon = np.linspace(0.094, 0.38, Settings.NUM_EPSILONS)
	rho = np.linspace(1e-9, 50, Settings.NUM_RHOS)

	if Settings.ENABLE_NUMERICAL_PLOT:
		# find and save the nucleation point for each epsilon
		for Settings.EPSILON in arrEpsilon:
			phi0, phi, dPhi, B = getConvergingPhi(rho)

			# Nucleation point if dPhi is max. Only look T the first half of phi to
			# ignore computational errors, which normally appears in the later half.
			maxIndex = 0

			for i in range(int(len(dPhi)/2)):
				if dPhi[i] > dPhi[maxIndex]:
					maxIndex = i

			arrR.append(rho[maxIndex])
			arrB.append(getConvergingB(rho, B))

			# print progress because it is slow
			print("{0:1.0f}%".format(100 * (Settings.EPSILON-0.094)/(0.38-0.094)))

			if Settings.ENABLE_DEBUGGING:
				plotAndSaveB_X(rho, B)

	# plot R-epsilon and save as a file
	plotAndSaveR_Epsilon(arrEpsilon, arrR)
	plotAndSaveB_Epsilon(arrEpsilon, arrB)

	print("[ode.py] (R-epsilon and B-epsilon saved in {0:s})".format(os.getcwd()))


class Settings:
	""" A class containing calculation settings and formatting options. The
		objects shown below are initialised to the default options.

		To change settings, copy the objects into main() and re-assign new
		settings.
	"""

	# calculation settings
	EPSILON = 0.2 # works in the range [0.094, 0.380]
	NUM_RHOS = 10000
	NUM_EPSILONS = 30

	# plot settings
	GRAPH_COLOUR = 'black'

	ENABLE_ANALYTIC_PLOT = True
	ENABLE_NUMERICAL_PLOT = True

	ANALYTIC_R_EPSILON_RANGE = [0.05, 0.38]
	ANALYTIC_B_EPSILON_RANGE = [0.05, 0.38]

	LINE_WIDTH = 1
	DOTTED_LINE_WIDTH = 0.5
	MARKER_SIZE = 3

	LINE_STYLE = '-'
	DOTTED_LINE_STYLE = ':'
	MARKER_STYLE = 'o'

	# other settings
	ENABLE_DEBUGGING = False

def main():
	""" """
	global Settings

	# settings for V-Rho and B-X plots:
	Settings.GRAPH_COLOUR = 'red'
	Settings.EPSILON = 0.13

	# solveForSingleEpsilon()

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

