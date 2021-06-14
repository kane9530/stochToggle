import numpy as np
from scipy.optimize import fsolve
from itertools import groupby
from toggle_equations import *
from math import *

def steady_states (alpha1, alpha2, a, c, b, d, lambda1, lambda2, myPara = None):
    """ Arguments to change when changing the identity of the parameter being changed:
    Line15: Set mypara[val] to the correct parameters
    Line19: Adjust the values in x_list
    Line43: Change the arguments into toggle_rhs
    """
    # Define step and grid size
    steps = 5
    highest_value = 100
    tol = 10**(-2) #10**(-8)
    steady_states_final = [] # Holds the steady states of all parameters
    steady_states = [] # Holds the steady states of current parameter value
    
    if myPara is None:
        for i in range(0, highest_value, steps):
            for j in range(0, highest_value, steps):
                x=fsolve(toggle_rhs, np.array([i*0.01,j*0.01]), args = (alpha1, alpha2, a, c, b, d, lambda1, lambda2))
                x_list = np.array([alpha1, alpha2, x[0], x[1]]) #Extract the solution from fsolve
                if steady_states == []:
                    steady_states.append(x_list) # So that we have an initial reference steady state to compare against subsequently.
                else:
                    for k in range (len(steady_states)):
                        ss = steady_states[k] # Holds the value of the (k+1)th steady state                        
                        if sqrt( pow( float( x_list[2] ) - float( ss[2] ), 2 ) + pow(  float( x_list[3] ) - float( ss[3] ), 2) ) < tol :                             #Tolerance condition to define 'similarity'
                            break #break out of this for loop entirely if the current x_list steady state matches any previous steady state.

                        elif k == len(steady_states)-1: #If we reach the end of the loop, i.e. traverse the entire list of steady states
                            steady_states.append(x_list)


        steady_states_final.append(steady_states)
        steady_states = []

        steady_state_tol = 10**(-10)
        x_coordinate_ss = []
        y_coordinate_ss = []

        for i in range (len(steady_states_final)):
            for j in range(len(steady_states_final[i])):
                ss = steady_states_final[i][j]
                current_steady_state = [ss[2], ss[3]]
                [dx_ss,dy_ss] =  toggle_rhs(current_steady_state,  ss[0], ss[1], a, c, b, d, lambda1, lambda2 )

                if ((abs(dx_ss) < steady_state_tol)==True) and ((abs(dy_ss) < steady_state_tol)==True):
                    x_coordinate_ss.append([ss[0], ss[1], ss[2]])
                    y_coordinate_ss.append([ss[0], ss[1], ss[3]])
        #print(x_coordinate_ss)
        #print(y_coordinate_ss)
        
    
    else: 
        for val in range(len(myPara)):
            alpha1 = myPara[val][0]
            alpha2 = myPara[val][1]
            for i in range(0, highest_value, steps):
                for j in range(0, highest_value, steps):
                    x=fsolve(toggle_rhs, np.array([i*0.01,j*0.01]), args = (alpha1, alpha2, a, c, b, d, lambda1, lambda2))
                    x_list = np.array([alpha1, alpha2, x[0], x[1]]) 
                    if steady_states == []:
                        steady_states.append(x_list)
                    else:
                        for k in range (len(steady_states)):
                            ss = steady_states[k]                        
                            if sqrt( pow( float( x_list[2] ) - float( ss[2] ), 2 ) + pow(  float( x_list[3] ) - float( ss[3] ), 2) ) < tol : 
                                break 

                            elif k == len(steady_states)-1: 
                                steady_states.append(x_list)


            steady_states_final.append(steady_states)
            steady_states = []

        steady_state_tol = 10**(-10)
        x_coordinate_ss = []
        y_coordinate_ss = []

        for i in range (len(steady_states_final)):
            for j in range(len(steady_states_final[i])):
                ss = steady_states_final[i][j]
                current_steady_state = [ss[2], ss[3]]
                [dx_ss,dy_ss] =  toggle_rhs(current_steady_state,  ss[0], ss[1], a, c, b, d, lambda1, lambda2 )

                if ((abs(dx_ss) < steady_state_tol)==True) and ((abs(dy_ss) < steady_state_tol)==True):
                    x_coordinate_ss.append([ss[0], ss[1], ss[2]])
                    y_coordinate_ss.append([ss[0], ss[1], ss[3]])
        #print(x_coordinate_ss)
        #print(y_coordinate_ss)

    return [x_coordinate_ss, y_coordinate_ss]


def stability (alpha1, alpha2, a, c, b, d, lambda1, lambda2, x_coordinate_ss, y_coordinate_ss):
    """ Arguments to change when changing the identity of the parameter being changed:
    Lines9-10: Change the parameter identity being changed"""
    
    stable_steady_states = []
    unstable_steady_states = []

    for i in range(0, len(x_coordinate_ss)):
        alpha1 = x_coordinate_ss[i][0]
        alpha2 = x_coordinate_ss[i][1]
        x_ss = x_coordinate_ss[i][2]
        y_ss = y_coordinate_ss[i][2]

        f_x = ( ( 4 * (x_ss**3) * (a**4) * ( 1 / (a**4 + x_ss**4 )**2 ) ) * ( b**4 / ( b**4 +  y_ss**4 ) ) ) -lambda1

        f_y = ( alpha1 + ( x_ss**4 / (a**4 +x_ss**4 ) ) ) * ( -4 * (b**4) * ( y_ss**3) / ( b**4 +  y_ss**4)**2 )

        g_y = ( ( 4 * (y_ss**3) * (c**4) * ( 1 / ( c**4 + y_ss**4 )**2 ) ) * ( d**4 / ( d**4 + x_ss**4 ) ) ) -lambda2

        g_x = ( alpha2 + ( y_ss**4 / (c**4 + y_ss**4 ) ) ) * ( -4 * (d**4) * (x_ss**3) / ( d**4 + x_ss**4)**2 )

        det = (f_x * g_y) - (f_y * g_x)

        if det < 0:
            unstable_ss_coord = [x_coordinate_ss[i], y_coordinate_ss[i]] #unstable
            unstable_steady_states.append(unstable_ss_coord)
        else:
            stable_ss_coord = [x_coordinate_ss[i], y_coordinate_ss[i]] #unstable
            stable_steady_states.append(stable_ss_coord)
    
    #print("mystable", stable_steady_states)
    #print("unstable", unstable_steady_states)
    
    return [stable_steady_states, unstable_steady_states]

def group_steady_states(stable_steady_states, unstable_steady_states, myPara):
    grouped_stable_ss = [[] for i in range(len(myPara))]
    grouped_unstable_ss = [[] for i in range(len(myPara))]

    counter = 0
    for key, group in groupby(stable_steady_states, lambda x: x[0][0:2]):
        for thing in group:
            grouped_stable_ss[counter].append(thing)
        counter+=1
        
    grouped_stable_ss = np.array([np.array(xi) for xi in grouped_stable_ss])
    
    counter = 0
    for key, group in groupby(unstable_steady_states, lambda x: x[0][0:2]):
        for thing in group:
            grouped_unstable_ss[counter].append(thing)
        counter+=1
    grouped_unstable_ss = np.array([np.array(xi) for xi in grouped_unstable_ss])
    
    return [grouped_stable_ss, grouped_unstable_ss]
    
    
