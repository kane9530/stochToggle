import os
import sys
import time 
import emcee
import numpy as np
import pandas as pd
import math
from scipy.integrate import odeint
import matplotlib.pyplot as plt
from multiprocessing import Pool
from multiprocessing import cpu_count 
import pickle

from toggle_equations import *
from attractor_stats import *
from steady_states import *
from sim_funcs import *
from plotting_funcs import *

### 1) Importing data ###
df = pd.read_csv("files/csv/mcmc_initialise.csv")
data = np.load("files/npy/loess_fit_vals_050421.npy")
nmp_numbers = len(df)
nmp_list = df[['sox2_norm', 'tbxta_norm']].values.tolist()
nmp_list_tcf = df[['sox2_norm', 'tbxta_norm','tcf_norm']].values.tolist()

### 2) Defining parameters for GRN model ###
num_ss = 100
dt = .1  # Time step.
T = 10  # Total time.
tolerance = 1e-8
max1 = 1
max2 = 1
n = int(T / dt)
t = np.linspace(0., T, n)  # Vector of times.
sim_stages = np.linspace(0,n-1,num_ss, dtype = int)

### 3) Define MCMC functions ###

def stoch_model(alpha1, alpha2, a, c, b, d, lambda1, lambda2, omega, exit_nmp_val):
    batch_num = 0
    batch = 3
    batch_nmp_num = [[] for i in range(batch)]

    while batch_num < batch:
        trajs_list = []
        i = 0
        while i < nmp_numbers:
            xinit = nmp_list[i][0]
            yinit = nmp_list[i][1]
            single_cell_sim = em_cle(1, xinit, yinit, tolerance, dt , T, alpha1, alpha2, a, c, b, d, max1, max2, lambda1, lambda2, omega)
            trajs_list.append(single_cell_sim)
            i+=1
        print("Embryo simulation complete: {} cells".format(nmp_numbers))

        num_nmps_time = []
        for i in range(num_ss) :
            nmp_counter = 0
            traj = np.squeeze(np.array(trajs_list))[:,:,sim_stages[i]]
            for cell in traj:
                if not ((cell[0] <= exit_nmp_val and cell[1] >= exit_nmp_val) or (cell[0] >= exit_nmp_val and cell[1] <= exit_nmp_val)):
                    nmp_counter += 1
            num_nmps_time.append(nmp_counter)
        print("num_nmps_time", num_nmps_time)
        batch_nmp_num[batch_num].append(num_nmps_time)
        batch_num +=1
    return batch_nmp_num

def logposterior(theta,sigma, x):
    lp = logprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + loglikelihood(theta, sigma,x)

def loglikelihood(theta, sigma,t):
    alpha1, alpha2, a, c, b, d, lambda1, lambda2, omega, exit_nmp_val = theta
    
    batch_nmp_num = stoch_model(alpha1, alpha2, a, c, b, d, lambda1, lambda2, omega, exit_nmp_val)
    md = np.mean(batch_nmp_num, axis = 0)
    return -0.5*np.sum((((md - data)/sigma)**2)[0][5:])

def logprior(theta):
    alpha1, alpha2, a, c, b, d, lambda1, lambda2, omega, exit_nmp_val = theta
    if 0 < alpha1 < 1 and \
       0 < alpha2 < 1  and \
       0  < a < 1  and \
       0 < c <1  and \
       0  < b < 1  and \
       0  < d < 1  and \
       0 < lambda1 < 5 and \
       0  < lambda2 < 5  and \
       0.1  < omega < 1000  and \
       0 < exit_nmp_val < 0.6:
        lp = 0.
    else:
        lp = -np.inf
    return lp

### 4) Initialising parameters ###
Nens = 24 #Number of walkers

alpha1_ini = np.random.uniform(0,1, Nens) 
alpha2_ini = np.random.uniform(0,1, Nens) 
a_ini = np.random.uniform(0,1, Nens) 
c_ini = np.random.uniform(0,1, Nens) 
b_ini = np.random.uniform(0,1, Nens) 
d_ini = np.random.uniform(0,1, Nens) 
lambda1_ini = np.random.uniform(0,5, Nens) 
lambda2_ini = np.random.uniform(0,5, Nens) 
omega_ini = np.random.uniform(10,1000, Nens) 
exit_nmp_val_ini = np.random.uniform(0,0.3, Nens) 

### 5) Define parameters for MCMC ###

inisamples = np.array([alpha1_ini, alpha2_ini, a_ini, \
                       c_ini, b_ini, d_ini,\
                       lambda1_ini, lambda2_ini, omega_ini,\
                       exit_nmp_val_ini]).T

ndims = inisamples.shape[1] 
Nburnin = 0  
Nsamples = 1000
dsigma = 0.25
argslist = (dsigma, sim_stages)
ncpu = cpu_count()
print("{0} CPUs".format(ncpu))

### 6) Running Ensemble sampler ###

with Pool() as pool:
    start = time.time()
    sampler = emcee.EnsembleSampler(Nens, ndims, logposterior, args = argslist, pool = pool, moves = emcee.moves.StretchMove(a = 1.3))
    sampler.run_mcmc(inisamples, Nsamples+Nburnin, progress=True)
    end = time.time()
    time_taken = end - start
    print("One run took {0:.1f} seconds".format(time_taken))
postsamples = sampler.chain[:, Nburnin:, :].reshape((-1, ndims))

### 7) Saving important outputs ###

samples = sampler.get_chain(flat=True)
print('samples')
print(samples.shape)
print(type(samples))
print(' ')
with open("samples_emcee.txt", "wb") as fp:   #Pickling
    pickle.dump(samples, fp)

log_probs = sampler.flatlnprobability
print('log_probs')
print(log_probs.shape)
print(type(log_probs))
print(' ')
with open("log_probs.txt", "wb") as fp:   #Pickling
    pickle.dump(log_probs, fp)
    
map_params = samples[np.argmax(sampler.flatlnprobability)]
print('map_params')
print(map_params.shape)
print(type(map_params))
print(' ')
with open("map_params_emcee.txt", "wb") as fp:   #Pickling
    pickle.dump(map_params, fp)
    
acceptance_fraction = sampler.acceptance_fraction
print('acceptance_fraction')
print(acceptance_fraction.shape)
print(type(acceptance_fraction))
print(' ')
with open("acceptance_fraction_emcee.txt", "wb") as fp:   #Pickling
    pickle.dump(acceptance_fraction, fp)
    
autocorr_time = sampler.get_autocorr_time(quiet=True)
print('autocorr_time')
print(autocorr_time.shape)
print(type(autocorr_time))
print(' ')
with open("autocorr_time_emcee.txt", "wb") as fp:   #Pickling
    pickle.dump(autocorr_time, fp)
