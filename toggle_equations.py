import numpy as np

def protein_production (x, alpha1, alpha2, a, c, b, d, max1, max2, tolerance = 1e-8): 
    """Return the cis-regulatory function i.e. the protein production rate.
    
    Keyword arguments: Definition and (units)
    x      -- input numpy array of concentration of 2 gene products X and Y. (concentration)
    alpha1 -- Rate of production of gene X, in absence of activation by X. (concentration/time)
    alpha2 -- Rate of production of gene Y, in absence of activation by Y. (concentration/time)
    a      -- Concentration for half-maximal activation by gene X. (concentration)
    c      -- Concentration for half-maximal activation by gene Y.(concentration)
    b      -- Concentration for half-maximal repression by gene Y.(concentration)
    d      -- Concentration for half-maximal repression by gene Y. (concentration)
    max1   -- Maximum rate of protein production caused by activator X.(concentration/time)
    max2   -- Maximum rate of protein production caused by activator Y.(concentration/time)"""
  
    xp,yp = x[0],x[1]
    xdot = (alpha1 +  max1*(xp**4 / (a**4+xp**4 )))*(b**4/ (b**4 +yp**4))
    ydot = (alpha2 + max2*(yp**4 / (c**4+yp**4 )))*(d**4/ (d**4 + xp**4)) 
    f = np.array([xdot, ydot])
    
    for conc in f:
        if conc < tolerance:
            conc = 0
    return f

def decay (x, lambda1, lambda2, tolerance = 1e-8):
    """Return the decay rate.
    
    Keyword arguments:
    lambda1-- decay rate of protein product X. (concentration/time)
    lambda2-- decay rate of protein product Y. (concentration/time) """
    
    xd,yd = x[0],x[1]
    xdot = lambda1*xd
    ydot = lambda2*yd
    f = np.array([xdot, ydot])
   
    for conc in f:
        if conc < tolerance:
            conc = 0
        
    return f
    
def deterministic_term (protein_production, decay):
    """Return the deterministic/mean function of the chemical langevin equation (CLE).
    
    This is equivalent to the Reaction Rate Equation determined from the law of mass action in 
    classic chemical kinetics""" 
  
    xp,yp,xd,yd = protein_production[0] , protein_production[1], decay[0], decay[1]
    xdot = xp-xd
    ydot = yp-yd
    f = np.array([xdot, ydot])
    return f


def toggle_rhs(x, alpha1, alpha2, a, c, b, d, lambda1, lambda2):

    xdot=[0,0]

    xdot[0] = ( alpha1 + ( x[0]**4 / ( a**4+x[0]**4 ) ) )*(b**4/ ( b**4 +x[1]**4 ) ) - (lambda1*x[0]);
    
    xdot[1] = ( alpha2 + ( x[1]**4 / ( c**4+x[1]**4 ) ) )*(d**4/ ( d**4 + x[0]**4 ) ) - (lambda2*x[1]);

    f = xdot
    
    return f