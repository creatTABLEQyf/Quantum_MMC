
# Script for Ising model in ASE

import numpy as np
from numpy import *
from ase import *
from ase import Atoms
from copy import deepcopy
import matplotlib.pyplot as plt
from matplotlib.legend_handler import HandlerLine2D
import csv

List = []
Energy_sys = []
Magn_sys = []
Cv_sys = []
Sus_sys = []



with open('Data_0.csv', 'r') as f:
    reader = csv.reader(f)
    next(f)
    for row in reader:
       print(row)
       List.append(float(row[0]))
       Energy_sys.append(float(row[1]))
       Magn_sys.append(float(row[2]))
       Cv_sys.append(float(row[3]))
       Sus_sys.append(float(row[4]))

#print "\nList is : ",List1
#print "\nEnergy_sys is: ",Energy_sys1
#print "\nMagn_sys is: ",Magn_sys1
#print "\nCv_sys is: ",Cv_sys1
#print "\nSus_sys is: ",Sus_sys1

fig1 = plt.figure()
fig1.suptitle('Heat capacity of lattice', fontsize = 12)
plt.xlabel('Temperature in KbT')
plt.ylabel('Heat capacity')
line1, = plt.plot(List,Cv_sys, 'g^' ,linestyle = "-.")
fig1.savefig('Heat capacity.pdf', dpi = 250, bbox_inches= 'tight',pad_inches=0.4)
#        
fig2 = plt.figure()
fig2.suptitle('Energy of lattice', fontsize = 12)
plt.xlabel('Temperature in KbT')
plt.ylabel('Energy')
line1, = plt.plot(List,Energy_sys,  'bo' ,linestyle = "-.")
fig2.savefig('Energy.pdf', dpi = 250, bbox_inches= 'tight',pad_inches=0.4)  

fig3 = plt.figure()
fig3.suptitle('Magnetization of lattice', fontsize = 12)
plt.xlabel('Temperature in KbT')
plt.ylabel('Normalised magnetization, m')
line1, = plt.plot(List,Magn_sys,  'ro' ,linestyle = "-.")
fig3.savefig('Magnetization.pdf', dpi = 250, bbox_inches= 'tight',pad_inches=0.4)

fig4 = plt.figure()
fig4.suptitle('Magnetic susceptibility', fontsize = 12)
plt.xlabel('Temperature in KbT')
plt.ylabel('Magnetic susceptibility')
line1, = plt.plot(List,Sus_sys, 'y^' ,linestyle = "-.")
fig4.savefig('Susceptibility_Magnetic.pdf', dpi = 250, bbox_inches= 'tight',pad_inches=0.4)


