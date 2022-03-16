"""

All mathematical variables used are in fact the "tilde-variables"
(e.g. V-tilde not V, phi-tilde not phi)

"""

import os
import numpy as np
from scipy.integrate import odeint
import matplotlib.pylab as plt

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

def getConvergingPhi(phiRangeToCheck, rho, epsilon):
	""" Determine initial phi that makes phi converge to a local maximum
		in potential V using binary search and tthe shooting method """

	def isDiverging(phi):
		return max(phi) > 2 or min(phi) < -2

	def isOscillating(phi):
		for i in range(len(phi)-1):
			if phi[i] > phi[i+1]:
				return True

		return False

	#
	minPhi0 = min(phiRangeToCheck)
	maxPhi0 = max(phiRangeToCheck)
	middlePhi0 = (minPhi0 + maxPhi0) / 2

	dPhi0 = 0
	lastConvergingPhi0 = None

	phi = odeint(ddPhi, [middlePhi0, dPhi0], rho, args=(epsilon, 0))[:, 0]

	# binary search for finding phi0
	for i in range(100):

		if isDiverging(phi):
			minPhi0 = middlePhi0

		elif isOscillating(phi):
			lastConvergingPhi0 = middlePhi0
			maxPhi0 = middlePhi0

		middlePhi0 = (minPhi0 + maxPhi0) / 2
		phi = odeint(ddPhi, [middlePhi0, dPhi0], rho, args=(epsilon, 0))[:, 0]

	# set phi to the last non-diverging solution if the last choice of phi0 still diverges
	# if isDiverging(phi) and lastConvergingPhi0 is not None:
		# phi = odeint(ddPhi, [lastConvergingPhi0, dPhi0], rho, args=(epsilon, 0))[:, 0]

	# return phi_initial and phi
	return middlePhi0, phi

def plotAndSavePotential(phiRange, epsilon):
	""" Plots potential versus phi. Used to find a suitable value for epsilon. """
	x = np.linspace(-1.5, 1.5, 1000)
	y = -V(x, epsilon)

	plt.clf()
	plt.title(r'Potential')
	plt.plot(x, y, color='red')

	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axvline(x=phiRange[0], color='black', linewidth=0.3, linestyle='--')
	plt.axvline(x=0, color='black', linewidth=0.5)
	plt.axvline(x=phiRange[1], color='black', linewidth=0.3, linestyle='--')

	plt.axis([-1.5, 1.5, 1.1*min(y), 1.1*max(y)])
	plt.xlabel(r'$\~{\phi}$', fontsize=15)
	plt.ylabel(r'$-\~V$', fontsize=15)

	plt.savefig('v_vs_phi.png', format='png', dpi=350)
	# plt.show()

def plotAndSavePhi(y, x, epsilon):
	""" Displays and saves the phi-rho plot generated """
	plt.clf()
	plt.title("Solution to Differential Equation", fontsize=15)
	plt.plot(x, y, color='red')

	plt.axhline(y=1, color='black', linewidth=0.3, linestyle='--')
	plt.axhline(y=0, color='black', linewidth=0.5)
	plt.axhline(y=-1, color='black', linewidth=0.3, linestyle='--')

	plt.axis([0, max(x), -1.5, 1.5])
	plt.xlabel(r'$\~{\rho}$', fontsize=15)
	plt.ylabel(r'$\~{\phi}$', fontsize=15)

	plt.savefig('phi_vs_rho.png', format='png', dpi=350)
	# plt.show()

def main():
	# initialise varibles
	epsilon = 0.38
	phiRange = [-1.2, 0]
	rho = np.linspace(1e-6, 50, 10000)

	# solve ODE
	phi0, phi = getConvergingPhi(phiRange, rho, epsilon)

	# print results
	plotAndSavePotential(phiRange, epsilon)
	plotAndSavePhi(phi, rho, epsilon)

	print("-" * 30)
	print("phi_initial = {0:f}".format(phi0))
	print("(figures saved in {0:s})".format(os.getcwd()))
	print("-" * 30)

if __name__ == "__main__":
	main()

