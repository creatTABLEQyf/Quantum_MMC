import qiskit
from qiskit import assemble, QuantumCircuit
from qiskit.visualization import plot_histogram
from qiskit.compiler import transpile
from random import *
from qiskit import Aer
sim = Aer.get_backend("statevector_simulator")
import numpy as np
import pandas as pd
from qiskit.circuit.library.standard_gates import HGate
from qiskit.circuit.library import RZGate, RYGate
import matplotlib.pyplot as plt
import multiprocessing as mp
import time

def neighbor_list(i, j, N):
  a = i-1
  b = j-1
  c = i+1
  d = j+1
  if a < 0:
      a = N-1
  if b < 0:
      b = N-1
  if c > N-1:
      c = 0
  if d > N-1:
      d = 0
  left   = (a, j)
  above  = (i, b)
  right  = (c, j)
  below  = (i, d)
  return [left, above, right, below]

spin, N = 16, 4

def single_step_qc(i, temp):
    global N, spin
    prob1= np.exp(-8/temp)
    angle1 = 2*np.arccos(np.sqrt(prob1))
    prob2 = np.exp(-4/temp)
    angle2 = 2*np.arccos(np.sqrt(prob2))
    ### construct neighbor qubit
    ising = np.arange(0, N**2).reshape(N,N)
    index = np.where(i==ising)
    neighbor_qubit = []
    for k in neighbor_list(index[0], index[1],N):
        neighbor_qubit.append(ising[k][0])
    ### construct circuit
    qc = QuantumCircuit(spin+4)
    aux_qubit = [spin, spin+1, spin+2, spin+3]
    # flip first
    qc.x(i)
    qc.barrier()
    for j in range(4):
        qc.cx(i, aux_qubit[j])
        qc.cx(neighbor_qubit[j], aux_qubit[j])
    # rotate or not depends on sign
    #####################situation 1 : all same signs #####################
    [aux1, aux2, aux3, aux4] = aux_qubit
    # default is ferromagnetic
    mcry= RYGate(angle1).control(4,label=None )
    qc.append(mcry,[aux1, aux2, aux3, aux4, i])
    #####################situation 2 : 3 same signs, 1 different signs #####################
    mcry_1= RYGate(angle2).control(4,label=None )
    mcry_2= RYGate(angle2).control(4,label=None )
    mcry_3= RYGate(angle2).control(4,label=None )
    mcry_4= RYGate(angle2).control(4,label=None )
    mcry_1._ctrl_state=14
    mcry_2._ctrl_state=13
    mcry_3._ctrl_state=11
    mcry_4._ctrl_state=7
    qc.append(mcry_1, [aux1, aux2, aux3, aux4, i])
    qc.append(mcry_2, [aux1, aux2, aux3, aux4, i])
    qc.append(mcry_3, [aux1, aux2, aux3, aux4, i])
    qc.append(mcry_4, [aux1, aux2, aux3, aux4, i])
    qc.barrier()
    qc.measure_all
    return qc

def cal_M(state):
    M = 0
    for i in state:
        if i == '1':
            M += 1
        else:
            M -= 1
    return M/16

def phase_space_sweep(temp, circuit_size = 32):
    # MC_sweep
    qc = QuantumCircuit(spin+4)
    # for 4x4 ising model at least need to repeat 16 circuits to get all basis (2**16)
    # I used 32 to make sure that we can cover the whole phase space
    # could change this number
    for _ in range(circuit_size):
        pick = randint(0, spin-1)
        qc.compose(single_step_qc(pick, temp), inplace=True)
    return qc

def measure_state(qc, sim):
    qc = transpile(qc,sim)
    final_counts = sim.run(qc, shots=8192).result().get_counts()
    prob, state = max(final_counts.values()), max(final_counts, key = final_counts.get)
    mag = abs(cal_M(state))
    return prob, 'st:'+state, mag

def measure(circuit):
    circuit.remove_final_measurements()
    circuit.measure_all()
    backend = Aer.get_backend('statevector_simulator')
    circuit = transpile(circuit, backend )
    final_counts = backend.run(circuit, shots=1).result().get_counts()
    print(final_counts)
    state = max(final_counts, key= final_counts.get)[4:]
    mag = abs(cal_M(state))
    print("Most probably state: ", state)
    print("Magnetization: ", mag)
    return state, mag

qc = phase_space_sweep(4)
superpositions = measure_state(qc, sim)
f_prob, f_state = max(superpositions.values()), max(superpositions, key = superpositions.get)
print(f_prob)
print(f_state)

# Average of many results from circuit size 32
temps = np.linspace(0.01, 15, 3)
df = pd.DataFrame()

def temp_sweep(temps):
	for temp in temps:
		magnetization = []
		for i in range(10):
			qc = phase_space_sweep(temp)
			prob, state, mag = measure_state(qc, sim)
			magnetization.append(mag)
		df['kT='+str(temp)] = magnetization

p = mp.Process(target=temp_sweep, args=(temps,))

df.to_csv('Circuit 32_Samples 10_statevector shots=8192, with measurements.csv')