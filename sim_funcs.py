import numpy as np
from toggle_equations import *
from steady_states import *
from plotting_funcs import *
import matplotlib.pyplot as plt

def em_cle (n_traj, x_init, y_init, tolerance, dt , T, alpha1, alpha2, a, c, b, d, max1, max2, lambda1, lambda2, omega, seed = 20):
    np.random.seed(seed)
    n = int(T / dt)
    t = np.linspace(0., T, n)
    sqrtdt =  np.sqrt(dt/omega) 
    x = np.zeros((n_traj,2,n))
    x[:,:,0]= [x_init,y_init]
    
    # Euler-Marayama method
    for k in range(n_traj):
        #print("{}/{} complete".format(k+1,n_traj))
        for i in range(n - 1):
            pp = protein_production([x[k][0][i], x[k][1][i]],alpha1,alpha2,a,c,b,d, max1, max2)
            de = decay([x[k][0][i], x[k][1][i]],lambda1,lambda2)
            detT = deterministic_term(pp,de)
            stochastic_term = np.array([np.sqrt(pp[0] + de[0]), np.sqrt(pp[1] + de[1])])
            x[k][0][i + 1] = x[k][0][i] + dt * detT[0] + stochastic_term[0] * sqrtdt * np.random.randn() #Chemical langevin equation
            x[k][1][i + 1] = x[k][1][i] + dt * detT[1] + stochastic_term[1] * sqrtdt * np.random.randn()

            # Tolerance - setting small values and negative values to 0 to account for negative values from the standard normal
            if(x[k][0][i + 1] < tolerance):
                x[k][0][i + 1] = 0
            elif(x[k][1][i + 1] < tolerance):
                x[k][1][i + 1] = 0
    return x

def em_cle_change (n_traj, x_init, y_init, tolerance, dt , T, alpha1, alpha2, a, c, b, d, max1, max2, lambda1, lambda2, omega, min_val, max_val, step_change, snap_interval = 10000, seed = 20, displayFig = False, saveFig = False):
    np.random.seed(seed)
    n = int(T / dt)
    t = np.linspace(0., T, n)
    sqrtdt =  np.sqrt(dt/omega) 
    x = np.zeros((n_traj,2,n))
    x[:,:,0]= [x_init,y_init]
    diverging_time = n/3
    
    # Euler-Marayama method
    for k in range(n_traj):
        print("{}/{} complete".format(k+1,n_traj))
        alpha1 = min_val # Need to re-initialise the value of alpha1 after each trajectory as it has been incremented
        for i in range(n - 1):
            if i%step_change == 0 and (i > diverging_time and i < diverging_time*2) :
                alpha1 += (max_val-min_val)/((n/2)/step_change)
            elif i%step_change == 0 and i > diverging_time*2:
                alpha1 = max_val
            pp = protein_production([x[k][0][i], x[k][1][i]],alpha1,alpha2,a,c,b,d, max1, max2)
            de = decay([x[k][0][i], x[k][1][i]],lambda1,lambda2)
            detT = deterministic_term(pp,de)
            stochastic_term = np.array([np.sqrt(pp[0] + de[0]), np.sqrt(pp[1] + de[1])])
            x[k][0][i + 1] = x[k][0][i] + dt * detT[0] + stochastic_term[0] * sqrtdt * np.random.randn() #Chemical langevin equation
            x[k][1][i + 1] = x[k][1][i] + dt * detT[1] + stochastic_term[1] * sqrtdt * np.random.randn()

            # Tolerance - setting small values and negative values to 0 to account for negative values from the standard normal
            if(x[k][0][i + 1] < tolerance):
                x[k][0][i + 1] = 0
            elif(x[k][1][i + 1] < tolerance):
                x[k][1][i + 1] = 0
            #Periodically [defined by snap_interval], calculate the steady states, stability and save plot. 
            
            if i%snap_interval == 0 and displayFig:
                x_coordinate_ss, y_coordinate_ss = steady_states (alpha1, alpha2, a, c, b, d, lambda1, lambda2)
                stable_steady_states, unstable_steady_states = stability (alpha1, alpha2, a, c, b, d, lambda1, lambda2, x_coordinate_ss, y_coordinate_ss)
                fig, ax = plt.subplots(figsize = (8,6))
                ax.plot(np.array(stable_steady_states)[:,0][:,2], np.array(stable_steady_states)[:,1][:,2], '.', color = "#6600cc",  markersize= 20)
                x_p = x[k][0][:i]
                y_p = x[k][1][:i]
                lc = colorline(x_p, y_p, cmap=plt.cm.cividis, alpha = 0.7)
                cbar = plt.colorbar(lc, ticks=[0, 1])
                cbar.ax.set_yticklabels(['Start', 'End'])
                plt.xlim(-0.05, 1.5)
                plt.ylim(-0.05, 1.5)
                plt.tight_layout()
                
                if saveFig:
                    fig.savefig('plots_stochastic/non_autonomy/3_path{}_time{}.png'.format(k,i), bbox_inches='tight')
        
    return x

def em_cle_tcf_alpha1 (n_traj, x_init, y_init, tolerance, dt , T, alpha2, a, c, b, d, max1, max2, lambda1, lambda2, omega, tcf_init, tcf_group, seed = 20):
    np.random.seed(seed)
    n = int(T / dt)
    t = np.linspace(0., T, n)
    sqrtdt =  np.sqrt(dt/omega) 
    x = np.zeros((n_traj,2,n))
    x[:,:,0]= [x_init,y_init]
    
    # Euler-Marayama method
    for k in range(n_traj):
        if (tcf_group == 0):
            x_vals = [0, n]
            y_vals = [tcf_init, 0]
            coefs = np.polyfit(x_vals,y_vals,3)
            xf=np.linspace(0,n,n)
            alpha_vals = np.polyval(coefs, xf)
            
            for i in range(n - 1):
                pp = protein_production([x[k][0][i], x[k][1][i]],alpha_vals[i],alpha2,a,c,b,d, max1, max2)
                de = decay([x[k][0][i], x[k][1][i]],lambda1,lambda2)
                detT = deterministic_term(pp,de)
                stochastic_term = np.array([np.sqrt(pp[0] + de[0]), np.sqrt(pp[1] + de[1])])
                x[k][0][i + 1] = x[k][0][i] + dt * detT[0] + stochastic_term[0] * sqrtdt * np.random.randn() #Chemical langevin equation
                x[k][1][i + 1] = x[k][1][i] + dt * detT[1] + stochastic_term[1] * sqrtdt * np.random.randn()

                # Tolerance - setting small values and negative values to 0 to account for negative values from the standard normal
                if(x[k][0][i + 1] < tolerance):
                    x[k][0][i + 1] = 0
                elif(x[k][1][i + 1] < tolerance):
                    x[k][1][i + 1] = 0
                    
        elif (tcf_group == 1):
            x_vals = [0, n]
            y_vals = [tcf_init, 1]
            coefs = np.polyfit(x_vals,y_vals,3)
            xf=np.linspace(0,n,n)
            alpha_vals = np.polyval(coefs, xf)

            for i in range(n - 1):
                pp = protein_production([x[k][0][i], x[k][1][i]],alpha_vals[i],alpha2,a,c,b,d, max1, max2)
                de = decay([x[k][0][i], x[k][1][i]],lambda1,lambda2)
                detT = deterministic_term(pp,de)
                stochastic_term = np.array([np.sqrt(pp[0] + de[0]), np.sqrt(pp[1] + de[1])])
                x[k][0][i + 1] = x[k][0][i] + dt * detT[0] + stochastic_term[0] * sqrtdt * np.random.randn() #Chemical langevin equation
                x[k][1][i + 1] = x[k][1][i] + dt * detT[1] + stochastic_term[1] * sqrtdt * np.random.randn()

                # Tolerance - setting small values and negative values to 0 to account for negative values from the standard normal
                if(x[k][0][i + 1] < tolerance):
                    x[k][0][i + 1] = 0
                elif(x[k][1][i + 1] < tolerance):
                    x[k][1][i + 1] = 0
                        
    return x