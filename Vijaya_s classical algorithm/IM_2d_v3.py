import numpy as np
from numpy import *
from ase import *
from ase import Atoms
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
# from ase.visualize import view
# from ase.visualize.primiplotter import *
# from ase.lattice.cubic import SimpleCubic
# from ase.calculators.neighborlist import NeighborList
# from ase.io import write
# from ase.io.trajectory import PickleTrajectory
import timeit
import time
import sys
import os
import csv
import pandas as pd
import multiprocessing as mp

data = pd.DataFrame()

# sys.stdout = open('Outfile', 'w')


Element = 'Fe'  ### Define Element Type
a = 2.870  ### Lattice constant
N = 3  ### Size of Lattice = N*N
muB = 0  ### External MagneticField
q = 2  ### No. of magnetic states, for Ising modes, q=2
J = 1  ### Coupling constant
NN = 1  ### NN = 1 (1st nearest), NN = 2 (2nd nearest) ##
Itt = 500000

# sys.stdout = open('Out.log', 'a', 0)

print("\n")
print("\n")
print("+++++++++++++++++++ Start of Program ++++++++++++++++++")

now = time.strftime("%c")

print("\nCurrent date : " + time.strftime("%x"))
print("\nCurrent time : " + time.strftime("%X"))

if NN == 1:
	Radius = a  ###1st nearest neighbor
	NumNeighors = 6  ### Define number of neighbor, based on CrystalStrusture
# print Radius

elif NN == 2:
	Radius = np.round((a / np.sqrt(2.)), decimals=5)  ###2nd nearest neighbor
	NumNeighors = 8  ### Define number of neighbor, based on CrystalStrusture
# print Radius

Element1 = Atoms([Atom(str(Element), (0, 0, 0), magmom=1.0)])

cell = [(a, 0, 0),
		(0, a, 0),
		(0, 0, a)]

Element1.set_cell(cell, scale_atoms=True)

Lattice = Element1.repeat((N, N, 1))

Lattice.set_pbc(True)

# print(Lattice)

print("\nSize of Lattice is : ", len(Lattice))
print("\nElement is: ", Element)
print("\nLattice constant is: ", a)


def Qstate(q):
	if q % 2 == 0:
		Q = (np.linspace(-(q / 2.), (q / 2.), q + 1)).tolist()
		Q.remove(0.)
		return Q
	else:
		Q = (np.linspace(-(q / 2), (q / 2), q)).tolist()
		return Q


Q_state = Qstate(int(q))
print("\nQ state are: ", Q_state)


def Nbl():  ### Find the distance to get the neighbor list based on minimum radius criteria ##
	time_in = timeit.default_timer()
	i = 0
	j = 0
	NL = []
	Index = 0
	for i in (Lattice):
		ID = []
		for j in (Lattice):
			d = np.round(Lattice.get_distance(i.index, j.index, mic=True), decimals=5)
			if d == Radius:
				ID.append((i.index, j.index))
			j.index += 1
		i.index += 1
		NL.append(ID)

		while len(NL[Index]) == NumNeighors:
			Index += 1
			break
		else:
			continue
	time_out = timeit.default_timer()
	TotTime = time_out - time_in
	print('Total time for calculation of neighbors is :', TotTime)
	return NL


NbrList = Nbl()


# print ('\nLength of neighbor list is :',len(NbrList))
# print(NbrList)
# print('Distance is: ',Distance)


def SpinSelection(Trial, R_i):
	Startspin = float(Trial[R_i])

	NewSpin = random.choice(Q_state)
	while NewSpin == Startspin:
		NewSpin = random.choice(Q_state)

	Trial[R_i] = NewSpin
	print("Trial is: ", Trial)
	return Trial


def EnergyIM(Trial, R_i):  ## Kronecker sum

	# print "***********************************"

	Sum = 0
	NbrI = 0
	Atomlist = NbrList[R_i]
	# print ("Atom list is:", Atomlist)
	# Index_1 = Atomlist[NbrI][0]
	while NbrI in range(len(Atomlist)):
		# print(NbrI)
		# print(Atomlist[NbrI][1])
		Trial_Index = Atomlist[NbrI][1]
		Sum += Trial[Trial_Index]
		# print(Sum)
		NbrI += 1
	Energy = Sum * Trial[R_i]
	# print("Energy is:", Energy)
	return Energy


Energy_sys1 = []
Magn_sys1 = []
Cv_sys1 = []
Sus_sys1 = []
List1 = np.linspace(0.1, 6, 50)  ### Define the temperature range and interval

print("\n=============###Monte Carlo Calculation###==============")


def MonteCarlo(List1):
	for kT in List1:
		Spins = Lattice.get_initial_magnetic_moments()
		print("\nInitial configuration is: \n", Spins)
		for i in range(len(Spins)):
			Spins[
				i] = 1  # np.random.choice(Qstate)         	### Starting configuration is Ferromagnetic. Can be initialized with random spins

		time_in1 = timeit.default_timer()

		print("Value of kT is", kT)
		# print ("\nInitial configuration is: \n", Lattice_old)
		Ener = 0
		for R_i in range(len(Spins)):
			Ener1 = EnergyIM(Spins, R_i)
			Ener += Ener1
		E_old = -J * ((Ener / 2.) + (muB * np.sum(Spins)))

		Magn_old = (np.sum((Spins), dtype=float)) / (len(Spins))

		print("\nEnergy of the Starting configuration: ", E_old)
		print("\nMagnetisation of starting configuration: ", Magn_old)
		print("\nItteration is: ", Itt)
		#            print "\nNo. of Itteration cycle is : ", Itt
		#            print "\nNo. of Equilibration cycle is : ", Equilibration
		# time_in = timeit.default_timer()
		Energy_step = []
		Mag_step = []
		Count = 0

		# Youfu's Modification: taking all the data points.
		magnetization = []

		for steps in range(Itt):
			R_index = np.random.randint(1, len(Lattice), 1)
			# print ('R_i is:', R_index)
			R_i = R_index[0]
			Ene1 = EnergyIM(Spins, R_i) + (muB * Spins[R_i])
			# print(Ene1)
			Trial = deepcopy(Spins)

			Trial[R_i] = -1 * Spins[R_i]
			# print "Trial spins are :", Spins_New
			Ene2 = EnergyIM(Trial, R_i) + (muB * Trial[R_i])  ## Energy of new spin congiguration for index R_i ##

			delE = float(J * (Ene1 - Ene2))
			# print "delE is : ", delE
			rd_num = np.random.rand(1)
			if delE < 0:
				E_new = E_old + delE
				E_old = E_new
				Magn_old = Magn_old + 2 * ((Trial[R_i]) / (len(Spins)))
				Spins = Trial
			elif rd_num < np.exp(-(1. / kT) * (delE)):
				E_new = E_old + delE
				E_old = E_new
				# print "\nEnergy of the NEW configuration is: ",E_old
				Magn_old = Magn_old + 2 * ((Trial[R_i]) / (len(Spins)))
				Spins = Trial

			if steps > 49000:  ### Number of steps for equilibration
				Energy_step.append(E_old)
				Mag_step.append(Magn_old)

			magnetization.append(Magn_old)

		Ener = 0
		for R_i in range(len(Spins)):
			Ener1 = EnergyIM(Spins, R_i)
			Ener += Ener1
		E_check = -J * ((Ener / 2.) + (muB * np.sum(Spins)))
		Magn_check = (np.sum((Spins), dtype=float)) / (len(Spins))

		#            print "E_old", E_old
		#            print "E_check", E_check

		# print ("\n*************************************************************************\n")
		print("\nFinal Lattice: ", Spins)
		for R_i in range(len(Spins)):
			Ener1 = EnergyIM(Spins, R_i)
			Ener += Ener1
		E_sys = -J * ((Ener / 2.) + (muB * np.sum(Spins)))

		Mag_sys = (np.sum((Spins), dtype=float)) / (len(Spins))
		Ener = np.sum(Energy_step) / len(Energy_step)
		Magn = np.sum(Mag_step) / len(Mag_step)
		Cv = (np.var(Energy_step)) * (1. / (len(Spins)) * (1. / (kT ** 2)))
		Sus = (np.var(Mag_step)) / (kT)
		var = np.var(Energy_step)
		print("\nSize of Magnetisation array: ",
			  size(Mag_step))  ### Is equal to Itt-No. of equilibration steps --> Should be always crosschecked
		print("\nEnergy is: ", Ener)
		print("\nMagnetisation is: ", Magn)
		print("\nHeat capacity : ", Cv)
		print("\nSusceptibility : ", Sus, "Variance of Magnetisation: ", np.var(Mag_step))
		Energy_sys1.append(Ener)
		Magn_sys1.append(Magn)
		Cv_sys1.append(Cv)
		Sus_sys1.append(Sus)

		time_out1 = timeit.default_timer()
		Totaltime1 = time_out1 - time_in1
		#        for i in range(len(Spins)):
		#            Value = Spins[i]
		#            Lattice[i].magmom = Value
		#            view(Lattice)
		#        name='traj_'+str(muB)+'.extxyz'
		#        outfile=os.path.join(trajectorydir1,name)
		#        write(outfile,Lattice)
		# print ("\nFinal Lattice: ", Spins)
		print('\nTotal time for simulation is: ', Totaltime1)
		print("\n*************************************************************************\n")
		data["kT=" + str(kT)] = magnetization


data.to_csv('Magnetization data.csv')

start = time.time()
print(MonteCarlo(List1))
end = time.time()
print(end - start)

'''
if __name__ == "__main__":
	List1 = np.linspace(0.1, 6, 50)
	start = time.time()
	p = mp.Process(target=MonteCarlo, args=(List1,))
	p.start()
	p.join()
	end = time.time()
	print(end - start)
'''

"""
with open('Data_' + str(muB) + '.csv', 'w') as csvfile:
	fieldnames = ['kT', 'Energy', 'Magnetisation', 'Heat Capacity', 'Susceptibility']
	writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
	writer.writeheader()
	i = 0
	for i in range(len(List1)):
		writer.writerow({'kT': str(List1[i]), 'Energy': str(Energy_sys1[i]), 'Magnetisation': str(Magn_sys1[i]),
						 'Heat Capacity': str(Cv_sys1[i]), 'Susceptibility': str(Sus_sys1[i])})
		i += 1
"""
