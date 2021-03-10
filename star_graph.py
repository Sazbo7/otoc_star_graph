from __future__ import print_function, division
import sys,os
from quspin.operators import hamiltonian, quantum_operator # Hamiltonian and observables
from quspin.tools.measurements import obs_vs_time # t_dep measurements
import numpy as np # generic math functions
from numpy.random import ranf,seed # pseudo random numbers
from joblib import delayed,Parallel # parallelisation
from quspin.operators import exp_op # operators
from quspin.basis import spin_basis_general # spin basis constructor
from quspin.tools.measurements import ent_entropy # Entanglement Entropy
import os
from time import clock
import matplotlib.pyplot as plt
import scipy as sp
from qutip import *
from qutip.piqs import *
import pandas as pd
from scipy.sparse import load_npz, save_npz

os.environ['KMP_DUPLICATE_LIB_OK']='True'


def OTOC_t(Hamiltonian, V_op, W_op, initial_state="Haar", t_max = 15, t_step=100, t_init=0, basis=None, seed=None, L=12):

    ####### Prepare initial state #######
    #Find Hilbert space size
    hilbert_space = Hamiltonian.diagonal().shape[0];
    np.random.seed(seed=seed)

    init_psi = generate_initial_state(L, initial_state, seed);

    #Prepare Haar random state (COMMENT OUT IF QUTIP IS NOT INSTALLED)
    t = np.linspace(0, t_max, t_step);

    print(init_psi.size)

    #Make W(t)V(0)|psi>
    WV_psi = np.zeros([t_step, init_psi.size], dtype=complex)

    V_psi = V_op.dot(init_psi);
    V_psi_t = Hamiltonian.evolve(V_psi, 0.0, t);
    WV_psi_t = W_op.dot(V_psi_t);

    for time in range(t_step):
        WV_psi[time] = Hamiltonian.evolve(WV_psi_t.T[time], 0.0, -t[time]);

    #Make V(0)W(t)|psi>
    psi_t = Hamiltonian.evolve(init_psi, 0.0, t);
    W_psi_t = W_op.dot(psi_t);

    W_psi = np.zeros([t_step, init_psi.size], dtype=complex)
    for time in range(t_step):
        W_psi[time] = Hamiltonian.evolve(W_psi_t.T[time], 0.0, -t[time]);

    VW_psi = V_op.dot(W_psi.T);

    OTOC_t = np.zeros([t_step]);

    for time in range(t_step):
        OTOC_t[time] = np.real(np.dot(np.conj(VW_psi.T)[time], WV_psi[time]));

    return OTOC_t


def generate_initial_state(L, initial_state='Haar', seed=None):

    if initial_state == "Haar":
        init_psi = np.random.normal(size=2**L) + 1.j * np.random.normal(size=2**L)
        init_psi /= np.linalg.norm(init_psi)

    elif initial_state == "pol_z":
        init_psi = np.zeros([2**L]);
        init_psi[0] = 1;

    elif initial_state == "pol_x":
        init_psi = np.ones([2**L]);

    elif initial_state == "pol_y":
        psi_y = [1.0, 1.0j];
        init_psi = psi_y;
        for i in range(L-1):
            init_psi = np.kron(init_psi, psi_y);

    elif initial_state == "InfT":
        init_psi = np.diagonal([2**L]);

    else:
        print("ERROR: invalid initial state defined \n Possible options: 'Haar', 'pol_[u]', 'inft'.")

    assert(init_psi.shape == (2**L, ))
    return init_psi;


def run_OTOC(J_zz, J_z, J_x, couplings, initial_state="Haar", t_max = 15, t_step=100, t_init=0, basis=None, seed=None, L=12, periodic=False):

    values = np.zeros([len(couplings), L+1, t_steps]);
    basis=spin_basis_1d(L=L+1);

    if periodic:
        BC = 0;
    else:
        BC = 1;

    H_zz = [[J_zz,i,(i+1)%L] for i in range(L-BC)] # PBC
    H_z = [[J_z,i] for i in range(L)] # PBC
    H_x = [[J_x,i] for i in range(L)] # PBC

    for coupling in range(len(couplings)):
        J_xxc, J_zc, J_xc = couplings[coupling], J_z, J_x;

        H_xxc = [[J_xxc / np.sqrt(L),i,L] for i in range(L)]
        H_zc = [[J_zc,L]]
        H_xc = [[J_xc,L]]

        # define static and dynamics lists
        static=[["zz",H_zz],["z",H_z],["x",H_x],["zz",H_xxc],["z",H_zc],["x",H_xc]];
        #static=[["zz",H_zz],["z",H_z],["x",H_x]];
        dynamic=[];
        H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);
        static_0 = [[1.0,0]];

        static=[["z",static_0]];
        V_op_0 = hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);

        #array = [1, 5, 10, 11];
        for i in range(L):
        #for i in array:
            static_N = [[1.0,i]];
            static=[["z",static_N]];
            W_op_N = hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);

            values[coupling,i] = OTOC_t(H, V_op_0, W_op_N, initial_state=initial_state, t_max = t_max, t_step=t_step, t_init=t_init, basis=None, seed=seed)

    return t_max, t_init, values;



def run_entanglement_entropy(J_zz, J_z, J_x, couplings, initial_state="Haar", t_max = 15, t_step=100, t_init=0, basis=None, seed=None, L=12, periodic=False):

    entanglement_entropy = np.zeros([len(couplings), t_steps]);
    basis=spin_basis_1d(L=L+1);

    init_psi = generate_initial_state(L, initial_state=initial_state, seed=seed);
    t = np.linspace(t_init, t_max, t_steps);

    if periodic:
        BC = 0;
    else:
        BC = 1;

    H_zz = [[J_zz,i,(i+1)%L] for i in range(L-BC)] # PBC
    H_z = [[J_z,i] for i in range(L)] # PBC
    H_x = [[J_x,i] for i in range(L)] # PBC

    for coupling in range(len(couplings)):
        J_xxc, J_zc, J_xc = couplings[coupling], J_z, J_x;

        H_xxc = [[J_xxc / np.sqrt(L),i,L] for i in range(L)]
        H_zc = [[J_zc,L]]
        H_xc = [[J_xc,L]]

        # define static and dynamics lists
        static=[["zz",H_zz],["z",H_z],["x",H_x],["zz",H_xxc],["z",H_zc],["x",H_xc]];
        #static=[["zz",H_zz],["z",H_z],["x",H_x]];
        dynamic=[];
        H=hamiltonian(static,dynamic,dtype=np.float64,basis=basis,check_herm=False);
        psi_t = H.evolve(init_psi, t_init, t);
        for i in range(len(t)):
            entanglement_entropy[coupling, i] = basis.ent_entropy(psi_t.T[i],sub_sys_A=range((L+1)//2));

    return t_max, t_init, entanglement_entropy;
