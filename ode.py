"""

All mathematical variables used are the "tilde-variables"
(e.g. V-tilde not V, phi-tilde not phi)

"""

import os
import numpy as np
import matplotlib.pylab as plt
from math import pi
from scipy.integrate import odeint
from scipy.optimize import fsolve
from scipy.signal import argrelextrema


# renaming the function 'fsolve' to 'getZeroAround' for clarity.
# Takes a function and an x coordinate, returns x0 closest to x such that f(x0) = 0.
def getZero(of, around, epsilon): return fsolve(of, around, epsilon)[0];

def V(phi, epsilon):
	""" Returns an array of V = V(phi, epsilon) in the range of phi. """
	return (1/8)*(phi**2 - 1)**2 + (epsilon/2)*(phi - 1)

def dV_dPhi(phi, epsilon):
	""" Returns dV/dPhi in the range of phi. """
	return (phi**3 - phi + epsilon) / 2

def ddPhi(initialConditions, rho, epsilon, potentialShift):
	""" Returns phi'' in the range of rho. """
	phi = initialConditions[0]
	dPhi = initialConditions[1]
	ddPhi = -(3/rho) * dPhi + dV_dPhi(phi, epsilon)
	dB = 2*(pi**2) * (rho**3) * ((1/2)*(dPhi**2) + (1/8)*(phi**2 - 1)**2 + (epsilon/2)*(phi - 1) + potentialShift)
	
	return dPhi, ddPhi, dB


def getConvergingPhi(rho, epsilon):
	""" Determine initial phi that makes phi converge to a local maximum
		in potential V using binary search and the shooting method """

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
	minPhi0 = getZero(of=dV_dPhi, around=-1, epsilon=epsilon)
	maxPhi0 = getZero(of=dV_dPhi, around=0, epsilon=epsilon)
	middlePhi0 = (minPhi0 + maxPhi0) / 2

	# calculate potential shift to make V(phi-) = 0
	falseVacuumIndex = argrelextrema(V(rho, epsilon), np.less)[0][0]
	potentialShift = -1 * V(rho, epsilon)[falseVacuumIndex]

	# print(argrelextrema(V(rho, epsilon), np.less))
	# print(potentialShift)

	# odeint returns [[phi0, dPhi0, B0], [phi1, dPhi1, B1], ...]
	solution = odeint(ddPhi, [middlePhi0, 0, 0], rho, args=(epsilon, potentialShift))

	# binary search for finding phi0
	for i in range(100):

		if isPhiDivergent():
			minPhi0 = middlePhi0

		elif isPhiOscillatory():
			maxPhi0 = middlePhi0

		middlePhi0 = (minPhi0 + maxPhi0) / 2
		solution = odeint(ddPhi, [middlePhi0, 0, 0], rho, args=(epsilon, potentialShift))

		# print("\n\n\n\n\n")

	# return phi0, phi, dPhi and B
	return middlePhi0, solution[:, 0], solution[:, 1], solution[:, 2]

def getConvergingB(rho, B):
	""" Finds and returns a value B converges to by comparing local sum-squared-
		errors. Based on the assumption that the true converging point would have
		the smallest SSE around the point.

		The name of this function is misleading since this function 'getConvergingB'
		does something different to 'getConvergingPhi'. Let me know any better name """

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


def plotAndSavePotential(epsilon):
	x = np.linspace(-1.5, 1.5, 1000)
	y = -V(x, epsilon)

	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axvline(x=0, color='black', linewidth=0.5)
	plt.axvline(x=getZero(of=dV_dPhi, around=-1, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axvline(x=getZero(of=dV_dPhi, around=0, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axvline(x=getZero(of=dV_dPhi, around=1, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.plot(x, y, color='red')

	rho = np.linspace(1e-9, 50, 10000)
	falseVacuum = -1 * V(rho, epsilon)[argrelextrema(V(rho, epsilon), np.less)[0][0]]
	plt.axhline(y=falseVacuum, color='grey', linewidth=0.3, linestyle='--')

	plt.axis([min(x), max(x), min(y), 1.2*max(y)])
	plt.xlabel(r'$\~{\phi}$', fontsize=15)
	plt.ylabel(r'$-\~V$', fontsize=15)

	plt.savefig('v_vs_phi.png', format='png', dpi=350)

def plotAndSavePhi(x, y, epsilon):
	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=getZero(of=dV_dPhi, around=1, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axhline(y=getZero(of=dV_dPhi, around=0, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axhline(y=getZero(of=dV_dPhi, around=-1, epsilon=epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.plot(x, y, color='red')

	plt.axis([0, max(x), -1.5, 1.5])
	plt.xlabel(r'$\~{\rho}$', fontsize=15)
	plt.ylabel(r'$\~{\phi}$', fontsize=15)

	plt.savefig('phi_vs_rho.png', format='png', dpi=350)

def plotAndSaveR(x, y):
	plt.clf()
	plt.plot(x, y, color='red', linestyle='', marker='o', markersize=3)

	# plt.axis([0, 0.46, 0, 1.2*max(y)])
	plt.xlabel(r'$\~{\epsilon}$', fontsize=15)
	plt.ylabel(r'$R$', fontsize=15)

	x = np.linspace(0.05, 0.38, 100)
	y = 2 / x
	plt.plot(x, y, color='red', linestyle='-')

	plt.savefig('r_vs_epsilon.png', format='png', dpi=350)

def plotAndSaveB(x, y):
	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=getConvergingB(x, y), color='grey', linewidth=0.3, linestyle='--')
	plt.plot(x, y, color='red', linestyle='-')

	plt.axis([0, max(x), 1.2*min(y), 1.5*getConvergingB(x, y)])
	plt.xlabel(r'$x$', fontsize=15)
	plt.ylabel(r'$B$', fontsize=15)

	plt.savefig('b_vs_x.png', format='png', dpi=350)
	
def plotAndSaveBX(x, y):
	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.plot(x, y, color='red', linestyle='', marker='o', markersize=3)

	# plt.axis([0, 0.46, , 1.2*max(y)])
	plt.xlabel(r'$\~{\epsilon}$', fontsize=15)
	plt.ylabel(r'$\~{B}$', fontsize=15)

	# x = np.linspace(0.05, 0.38, 100)
	# y = 27 * (pi**2) * (2/3)**4 / (2 * x**3)
	# plt.plot(x, y, color='red', linestyle='-')

	# x = np.linspace(0.05, 0.38, 100)
	# y = 27 * (pi**2) * (1/3)**4 / (2 * x**3)
	# plt.plot(x, y, color='black', linestyle='-')

	plt.savefig('b_vs_epsilon.png', format='png', dpi=350)


def solveForSingleEpsilon():
	""" Solves the bubble equation to get phi0 and phi for a given epsilon.
		Also generates and saves phi-rho and V-phi plots. """

	# initialise varibles
	epsilon = 0.3 # works in the range [0.094, 0.38]
	rho = np.linspace(1e-9, 50, 10000)

	# solve ODE
	phi0, phi, dPhi, B = getConvergingPhi(rho, epsilon)

	# print results
	plotAndSavePotential(epsilon)
	plotAndSavePhi(rho, phi, epsilon)
	plotAndSaveB(rho, B)

	print("-" * 30)
	print("For epsilon = {0:f}, phi_initial = {1:f}".format(epsilon, phi0))
	print("(figures saved in {0:s})".format(os.getcwd()))
	print("-" * 30)

def solveForEpsilonArray():
	""" Solves the bubble equation for a given range of epsilon.
		Also generates and saves an R-versus-epsilon plot. """

	# initialise variables
	arrR = []
	arrB = []
	arrEpsilon = np.linspace(0.094, 0.38, 10)
	rho = np.linspace(1e-9, 50, 1000)

	# find and save the nucleation point for each epsilon
	for epsilon in arrEpsilon:
		phi0, phi, dPhi, B = getConvergingPhi(rho, epsilon)

		# Nucleation point if dPhi is max. Only look T the first half of phi to
		# ignore computational errors, which normally appears in the later half.
		maxIndex = 0

		for i in range(int(len(dPhi)/2)):
			if dPhi[i] > dPhi[maxIndex]:
				maxIndex = i

		arrR.append(rho[maxIndex])
		arrB.append(getConvergingB(rho, B))

		# print progress because it is slow
		print("{0:1.0f}%".format(100 * (epsilon-0.094)/(0.38-0.094)))

	# plot R-epsilon and save as a file
	plotAndSaveR(arrEpsilon, arrR)
	plotAndSaveBX(arrEpsilon, arrB)


def main():
	solveForSingleEpsilon()
	solveForEpsilonArray()

if __name__ == "__main__":
	main()

