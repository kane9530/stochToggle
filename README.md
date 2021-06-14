# stochToggle

## Basic Introduction

I describe the formulation of a stochastic, non-autonomous toggle switch to model the dynamics of a gene regulatory network.

A toggle switch consists of two genes that mutually repress each other, and in the formulation that I use, they are also capable of auto-activation [Verd et al., 2014](https://bmcsystbiol.biomedcentral.com/ articles/10.1186/1752-0509-8-43). The basic formulation of the toggle switch involves a pair of coupled ordinary differential equation (ODE) terms:

![ODEs](/images/hill_equations.png)

In the ODEs, the rate of change of mRNA levels is a function of a production term, modelled by a hill function, and a decay term. In theabsence of a stochastic term, the toggle switch is deterministic - multiple simulations carried out at the same initial conditions with the same parameter set will always yield the same output trajectory. In addition, only the state-variables (X and Y) change with time and all parameter values are held constant throughout the simulations - the formulation is autonomous. 

- <strong>toggle_equations.py</strong> python file contains the ODEs for the toggle switch
-  <strong>steady_states.py</strong> python file describes how the stability of the steady states are determined. Essentially, it involves a linear stability analysis and the examination of the jacobian.
- <strong> sim_funcs.py </strong> python file describes the simulation of the stochastic oggle switch with the Euler-Maruyama scheme.
- <strong> attractor_stats.py </strong> python file describes how I determined several important properties of the stochastic trajectories - first passage time, exit time etc.
- <strong> plotting_funcs.py </strong> contains two helper functions used in making some of the plots
- <strong> mcmc_tcf.py </strong> contains the code for the ensemble MCMC method implemented from the emcee package.

## Deterministic, autonomous toggle switch
In <strong> 01_deterministic.ipynb</strong>, I briefly explore the consequences of the deterministic, autonomous genetic toggle switch.

## Stochastic, autonomous toggle switch
However, gene expression occurs at the molecular level and in certain circumstances, stochasticity can play a significant role in the process. By converting the continuous deterministic ODE representation above into a stochastic representation written in the form of the Chemical Langevin Equation [Gillespie, 2000](https://aip.scitation.org/doi/10.1063/1.481811), I explore the effects of stochasticity in the toggle switch in <strong> 02_stochastic.ipynb </strong>. 

![CLE_equations](/images/CLE_equations.png)

## Stochastic, non-autonomous toggle switch

The above two notebooks explore the toggle switch using an autonomous formulation. In reality, during embryonic development, cells experience a constantly changing signalling environment. To reflect this, I incorporated explicit time-dependence to the the external activation parameters and explored the effects of non-autonomy in silico. In <strong> 03_nonAutonomy.ipynb </strong>, I discover that some cells 'lag' behind in the wrong attractor, providing a possible mechanistic basis for the existence of 'rebellious' cells in my experimental data.

## Curve fitting to experimental data

I sought to fit the computational models to the number of NMps at each developmental stage. I fitted a simple curve to the count data in <strong> 04_curve_fitting.ipynb </strong>

## Markov-chain Monte Carlo simulations 

Using the stochastic toggle switch model, I simulated several trajectories initialised from an 18 somite-stage zebrafish HCR image. I then examined the  gene regulatory networks that yielded a good fit to the data and explored the parameter values returned from the MCMC simulations. The results are still being finalised and are documented in <strong>05_MCMC_fitting_autonomous.ipynb</strong>  and <strong> 05b)MCMC_fitting_non-autonomous.ipynb </strong>.




