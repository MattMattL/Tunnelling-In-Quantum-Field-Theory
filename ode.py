"""

All mathematical variables used are the "tilde-variables"
(e.g. V-tilde not V, phi-tilde not phi)

"""

import os
import numpy as np
from scipy.integrate import odeint
from scipy.optimize import fsolve
import matplotlib.pylab as plt

# renaming the function 'fsolve' to 'getZeroAround'.
# Takes a function and an x coord. Returns x0 closest to x such that f(x) = 0.
def getZeroAround(x, function, epsilon): return fsolve(function, x, epsilon)[0];

def V(phi, epsilon):
	""" Returns an array of V = V(phi, epsilon) in the range of phi. """
	return (1.0/8)*(phi**2 - 1)**2 + (epsilon/2)*(phi-1)

def dV_dPhi(phi, epsilon):
	""" Returns dV/dPhi in the range of phi. """
	return (phi**3 - phi + epsilon) / 2

def ddPhi(initialConditions, rho, epsilon, dummy):
	""" Returns phi'' in the range of rho. """
	phi = initialConditions[0]
	dPhi = initialConditions[1]
	ddPhi = -(3/rho) * dPhi + dV_dPhi(phi, epsilon)
	
	return dPhi, ddPhi


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

	# set lower & upper bounds for phi0
	minPhi0 = getZeroAround(-1, dV_dPhi, epsilon)
	maxPhi0 = getZeroAround(0, dV_dPhi, epsilon)
	middlePhi0 = (minPhi0 + maxPhi0) / 2

	dPhi0 = 0
	lastConvergingPhi0 = None

	# odeint() returns [[phi0, dPhi0], [phi1, dPhi1], [phi2, dPhi2] ...]
	solution = odeint(ddPhi, [middlePhi0, dPhi0], rho, args=(epsilon, 0))

	# binary search for finding phi0
	for i in range(100):

		if isPhiDivergent():
			minPhi0 = middlePhi0

		elif isPhiOscillatory():
			lastConvergingPhi0 = middlePhi0
			maxPhi0 = middlePhi0

		middlePhi0 = (minPhi0 + maxPhi0) / 2
		solution = odeint(ddPhi, [middlePhi0, dPhi0], rho, args=(epsilon, 0))

	# set phi to the last non-diverging solution if the last choice of phi0 still diverges
	if isPhiDivergent() and lastConvergingPhi0 is not None:
		solution = odeint(ddPhi, [lastConvergingPhi0, dPhi0], rho, args=(epsilon, 0))

	# return phi_initial and phi
	return middlePhi0, solution[:, 0], solution[:, 1]


def plotAndSavePotential(epsilon):
	x = np.linspace(-1.5, 1.5, 1000)
	y = -V(x, epsilon)

	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axvline(x=0, color='black', linewidth=0.5)
	plt.axvline(x=getZeroAround(-1, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axvline(x=getZeroAround(0, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axvline(x=getZeroAround(1, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.plot(x, y, color='red')

	plt.axis([-1.5, 1.5, 1.1*min(y), 1.1*max(y)])
	plt.xlabel(r'$\~{\phi}$', fontsize=15)
	plt.ylabel(r'$-\~V$', fontsize=15)

	plt.savefig('v_vs_phi.png', format='png', dpi=350)
	# plt.show()

def plotAndSavePhi(x, y, epsilon):
	plt.clf()
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=getZeroAround(1, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axhline(y=getZeroAround(0, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.axhline(y=getZeroAround(-1, dV_dPhi, epsilon), color='grey', linewidth=0.3, linestyle='--')
	plt.plot(x, y, color='red')

	plt.axis([0, max(x), -1.5, 1.5])
	plt.xlabel(r'$\~{\rho}$', fontsize=15)
	plt.ylabel(r'$\~{\phi}$', fontsize=15)

	plt.savefig('phi_vs_rho.png', format='png', dpi=350)
	# plt.show()

def plotAndSaveR(x, y):
	plt.clf()
	plt.plot(x, y, color='red', linestyle='', marker='o', markersize=3)

	plt.axis([0, 0.46, 0, 1.2*max(y)])
	plt.xlabel(r'$\~{\epsilon}$', fontsize=15)
	plt.ylabel(r'$R$', fontsize=15)

	plt.savefig('r_vs_epsilon.png', format='png', dpi=350)


def solveForSingleEpsilon():
	""" Solves the bubble equation to get phi0 and phi for a given epsilon.
		Also generates and saves phi-rho and V-phi plots. """

	# initialise varibles
	epsilon = 0.2 # works in the range [0.094, 0.38]
	rho = np.linspace(1e-9, 50, 10000)

	# solve ODE
	phi0, phi, dPhi = getConvergingPhi(rho, epsilon)

	# print results
	plotAndSavePotential(epsilon)
	plotAndSavePhi(rho, phi, epsilon)

	print("-" * 30)
	print("For epsilon = {0:f}, phi_initial = {1:f}".format(epsilon, phi0))
	print("(figures saved in {0:s})".format(os.getcwd()))
	print("-" * 30)

def solveForEpsilonArray():
	""" Solves the bubble equation for a given range of epsilon.
		Also generates and saves an R-versus-epsilon plot. """

	# initialise variables
	arrR = []
	arrEpsilon = np.linspace(0.094, 0.38, 30)
	rho = np.linspace(1e-9, 50, 10000)

	# find and save the nucleation point for each epsilon
	for epsilon in arrEpsilon:
		phi0, phi, dPhi = getConvergingPhi(rho, epsilon)

		# "Nucleation point if dPhi is max". Only look the first half of phi to
		# ignore diverging lines for a large phi as they are computational errors.
		maxIndex = 0

		for i in range(int(len(dPhi)/2)):
			if dPhi[i] > dPhi[maxIndex]:
				maxIndex = i

		arrR.append(rho[maxIndex])

		# print progress because it is slow
		print("{0:4.0f}%".format( 100 * (epsilon-0.094)/(0.38-0.094) ))

	plotAndSaveR(arrEpsilon, arrR)


def main():
	# solveForSingleEpsilon()
	solveForEpsilonArray()

if __name__ == "__main__":
	main()

