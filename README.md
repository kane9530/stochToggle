# stochToggle

## Basic Introduction

I describe the formulation of a stochastic, non-autonomous toggle switch to model the dynamics of a gene regulatory network.

A toggle switch consists of two genes that mutually repress each other, and in the formulation that I use, they are also capable of auto-activation (See Verd et al., 2014). The basic formulation of the toggle switch involves a pair of coupled ODE terms:

![ODEs](/images/hill_equations.png)

In the ODE, the rate of change of mRNA levels is a function of a production term, modelled by a hill function, and a decay term. In the absence of a stochastic term, the toggle switch is deterministic - multiple simulations carried out at the same initial conditions with the same parameter set will always yield the same output trajectory. In addition, only the state-variables (X and Y) change with time and all parameter values are held constant throughout the simulations - the formulation is autonomous. In <strong> 01_deterministic.ipynb</strong>, I briefly explore the consequences of the deterministic, autonomous genetic toggle switch.



