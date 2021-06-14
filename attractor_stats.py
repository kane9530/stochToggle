import numpy as np
import numbers

def first_passage_time (traj_matrix, T, n, dist_x, dist_y, attractors_coord):
    """Return the first passage times of trajectories.
    
    Keyword arguments: Definition 
    traj_matrix -- trajectory array with dimensions (n_trajs, 2, n), where n_trajs is the
    number of computed trajectories and n is the number of timepoints. This is the output of the 
    euler_marayama_CLE function.
    T           -- total simulation duration
    n           -- number of time subintervals 
    dist_x      -- distance threshold for gene X to consider as having entered the steady state. 
    dist_y      -- distance threshold for gene Y to consider as having entered the steady state
    attractors_coord -- array with the coordinates of the stable steady states of interest.  """
    
    # We calculate the empirical first passage time using two methods: 
    # The next method returns the first (hence first passage) paired coordinates that match the euclidean distance condition. This is the first passage index.
    # np.nonzero returns the index of the first passage time coordinates as this is when the condition is True. This index is the time. Therefore, we have the first passage time.

    first_passage_time = np.zeros((traj_matrix.shape[0], len(attractors_coord)))
    for traj in range(traj_matrix.shape[0]):
        paired_coord = np.array(list(zip(traj_matrix[traj,0,:], traj_matrix[traj,1,:]))) 
        for index in range(len(attractors_coord)):
            fp_coord = next((x for x in paired_coord if (np.linalg.norm(x[0]- attractors_coord[index][0][2])) < dist_x and (np.linalg.norm(x[1]- attractors_coord[index][1][2])) < dist_y), -1)
            if not isinstance(fp_coord, int): #If there's no passage into the attractor, the next method returns the default int -1. 
                fp_time, _ = np.nonzero(paired_coord == fp_coord)
                first_passage_time[traj][index] = fp_time[0]/n * T #Converting the time index into simulation time
    return first_passage_time

def final_attractor(trajectories, dist_exit_x, dist_exit_y, attractors_coord):
    final_attractor = []

    if trajectories.shape[0] == 1:
        final_time_coord = [trajectories[0][-1], trajectories[1][-1]]
        for index in range(len(attractors_coord)):
            if (np.linalg.norm(final_time_coord[0]- attractors_coord[index][0][2])) < dist_exit_x and (np.linalg.norm(final_time_coord[1]- attractors_coord[index][1][2])) < dist_exit_y :
                final_attractor.append(index)
                break
            if index == len(attractors_coord)-1: 
                final_attractor.append(-1)
    
    else:
        for traj in trajectories:
            final_time_coord = [traj[0][-1], traj[1][-1]]
            for index in range(len(attractors_coord)):
                if (np.linalg.norm(final_time_coord[0]- attractors_coord[index][0][2])) < dist_exit_x and (np.linalg.norm(final_time_coord[1]- attractors_coord[index][1][2])) < dist_exit_y :
                    final_attractor.append(index)
                    break
                if index == len(attractors_coord)-1: 
                    final_attractor.append(-1)
    return final_attractor

def entry_time (paired_coord, dist_entry_x, dist_entry_y, attractor_coord, current_time):
    if current_time < (paired_coord.shape[0] -1):
        fp_coord = next((x for x in paired_coord[current_time:] if (np.linalg.norm(x[0]- attractor_coord[0][2])) < dist_entry_x and (np.linalg.norm(x[1]- attractor_coord[1][2])) < dist_entry_y), -1)
        fp_time = 0
        current_time = paired_coord.shape[0] -1
        if not isinstance(fp_coord, int):
            try:
                fp_time, _ = np.nonzero(paired_coord == fp_coord)[0]
                current_time = fp_time
            except ValueError:
                if fp_coord[0] == 0 and len(np.nonzero(paired_coord == fp_coord)[0]) != 1:
                    print("Fp_coord at 0 with multiple identical coordinates.\nAdjusted fp_time to take the first value")
                    fp_len = len(np.nonzero(paired_coord == fp_coord)[0])
                    #print(fp_len)
                    fp_time = np.nonzero(paired_coord == fp_coord)[0][fp_len-1]
                    current_time = fp_time 
    else:
        fp_time = 0
        current_time = paired_coord.shape[0] -1
        
            
    return fp_time, current_time


def exit_time (paired_coord, dist_exit_x, dist_exit_y, attractor_coord, current_time):
    if current_time < (paired_coord.shape[0] -1):
        fe_coord = next((x for x in paired_coord[current_time:]  if (np.linalg.norm(x[0]- attractor_coord[0][2])) > dist_exit_x or (np.linalg.norm(x[1]- attractor_coord[1][2])) > dist_exit_y), -1)
        #print("fe_coord", fe_coord)
        fe_time = 0
        current_time = paired_coord.shape[0] -1
        #print("Exit time_ fe coord", fe_coord)
        if not isinstance(fe_coord, int):
            try:
                fe_time, _ = np.nonzero(paired_coord == fe_coord)[0]
                current_time = fe_time 
            except ValueError:
                if fe_coord[0] == 0 and len(np.nonzero(paired_coord == fe_coord)[0]) != 1:
                    print("Fe_coord at 0 with multiple identical coordinates.\nAdjusted fe time to take the last value")
                    fe_len = len(np.nonzero(paired_coord == fe_coord)[0])
                    fe_time = np.nonzero(paired_coord == fe_coord)[0][fe_len-1]
                    current_time = fe_time 
                       
    else:
        fe_time = 0
        current_time = paired_coord.shape[0] -1
        
    return fe_time, current_time


def switch_times(traj_matrix, dist_entry_x, dist_entry_y, dist_exit_x, dist_exit_y, attractors_coord):
    all_times = [] #Contains a list of lists 
    for traj in range(traj_matrix.shape[0]):
        paired_coord = np.array(list(zip(traj_matrix[traj,0,:], traj_matrix[traj,1,:]))) 
        times = [] # Contains a list of lists of attractor entry and exit times for all attractors in trajectory x
        print("{}/{}".format(traj,traj_matrix.shape[0]))
        for index in range(len(attractors_coord)):
            #print ("entering attractor", index)
            current_time = 0
            attractor_times = [] # Contains the list of paired entry and exit times for attractor x 
            while current_time != (paired_coord.shape[0] -1) :
                fp_time, current_time = entry_time(paired_coord, dist_entry_x, dist_entry_y, attractors_coord[index], current_time)
               # print("enter current time", current_time)
                #print("the fp time", fp_time ,"the post-fp current_time", current_time)
                fe_time, current_time = exit_time(paired_coord,  dist_exit_x, dist_exit_y, attractors_coord[index], current_time)
               # print("the fe time", fe_time, "the post-fe current_time", current_time)
                attractor_times.append(np.array([fp_time, fe_time]))   

            #print("attractor_times", attractor_times)
            attractor_times_np = np.array(attractor_times)
            #print("attractor_times", attractor_times_np)
            #cumsum_attractor_times = np.reshape(np.cumsum(attractor_times_np, axis=0), attractor_times_np.shape)
            #print("cumsum", cumsum_attractor_times)
            #print(attractor_times_np)
            #print(cumsum_attractor_times)
            #times.append(cumsum_attractor_times)
            times.append(attractor_times_np)
        all_times.append(times)

    all_times = np.array([all_times]).squeeze()

    return all_times

            
def dwell_times(switch_times, total_sim_time, time_steps):
    sim_length = total_sim_time / time_steps
    all_dwell_times = [[] for _ in range(switch_times.shape[0])]
    mean_dwell_times = [[] for _ in range(switch_times.shape[0])]
    if switch_times.shape[0] == 0:
        raise TypeError ("No switching times were given as argument")
    else:
        for traj_index in range(switch_times.shape[0]):
            for attractor in range(switch_times.shape[1]):
                attractor_dwell_time = []
                switch_time = switch_times[traj_index][attractor]
                #unique_times = np.array(list(map(np.asarray, set(map(tuple, switch_times[traj_index][attractor]))))) # Removing duplicate times in the numpy array
                for switch in switch_time: 
                    if switch[0] == 0:
                        break
                    elif switch[1] == 0 : #If exit time = 0, means that system never exited attractor.  
                        dwell_time = sim_length - switch[0] # Trajectory stays in this attractor for the rest of sim time
                    else:
                        dwell_time = switch[1] - switch[0] #Dwell time is exit time - entry time
                    if dwell_time < 0: 
                        raise ValueError ("Negative dwell times obtained")
                    attractor_dwell_time.append(dwell_time)
                if len(attractor_dwell_time) > 0:
                    mean_time = sum(attractor_dwell_time) / len(attractor_dwell_time) 
                    mean_dwell_times[traj_index].append(mean_time)
                    all_dwell_times[traj_index].append(attractor_dwell_time)
                else:
                    mean_dwell_times[traj_index].append(0)
                    all_dwell_times[traj_index].append([0])
                
                          
    return (all_dwell_times, mean_dwell_times)

def switch_stats(switch_times, stable_steady_states):
    entry_times_tot, attractor_indices_tot = ([] for _ in range(2))
    if switch_times.ndim > 1: 
        for trajectory_times in switch_times: 
            entry_times, attractor_indices = ([] for _ in range(2)) 
            for attractor_index in range(len(trajectory_times)):
                if isinstance(trajectory_times[attractor_index].squeeze()[0], numbers.Integral) and trajectory_times[attractor_index].squeeze()[0] != 0 :
                    entry_times.append(trajectory_times[attractor_index].squeeze()[0])
                    attractor_indices.append(attractor_index)
                elif isinstance(trajectory_times[attractor_index].squeeze()[0], (list,np.ndarray)):
                    for attractor_time in trajectory_times[attractor_index].squeeze():
                        if attractor_time[0] != 0 :
                            entry_times.append(attractor_time[0])
                            attractor_indices.append(attractor_index)
            zipped_pairs = list(zip(entry_times, attractor_indices))
            sorted_attractor_indices = [x for _,x in sorted(zipped_pairs)]
            sorted_entry_times = sorted(entry_times)
            entry_times_tot.append(sorted_entry_times)
            attractor_indices_tot.append(sorted_attractor_indices)
        
        attractor_transition_mat = np.zeros((switch_times.shape[0], len(stable_steady_states),len(stable_steady_states)), dtype = np.int16)
        attractor_counts = np.zeros((switch_times.shape[0], len(stable_steady_states)), dtype = np.int16)
    
        for n_traj in range(switch_times.shape[0]):
            print("{}/{}".format(n_traj,switch_times.shape[0]))
            y = list(zip(entry_times_tot[n_traj],attractor_indices_tot[n_traj]))

            ## Counting number of times system enters each attractor
            for l in y:
                attractor_counts[n_traj,l[1]] += 1
                
            ## Creating the attractor_transition_matrix
            j = 0
            while j < len(y)-1:
                #print("In while loop", y[j][1])
                attractor_transition_mat[n_traj,(y[j])[1],(y[j+1])[1]] += 1
                j+=1
                
    else: #So we can also extract entry times for single trajectories (otherwise only collections allowed)
        trajectory_times = switch_times
        for attractor_index in range(len(trajectory_times)):
            if isinstance(trajectory_times[attractor_index].squeeze()[0], numbers.Integral) and trajectory_times[attractor_index].squeeze()[0] != 0 :
                entry_times_tot.append(trajectory_times[attractor_index].squeeze()[0])
                attractor_indices_tot.append(attractor_index)
            elif isinstance(trajectory_times[attractor_index].squeeze()[0], (list,np.ndarray)):
                for attractor_time in trajectory_times[attractor_index].squeeze():
                    if attractor_time[0] != 0 :
                        entry_times_tot.append(attractor_time[0])
                        attractor_indices_tot.append(attractor_index)      
        zipped_pairs = list(zip(entry_times_tot, attractor_indices_tot))
        
        attractor_indices_tot = [x for _,x in sorted(zipped_pairs)]
        entry_times_tot = sorted(entry_times_tot)
        attractor_transition_mat = np.zeros((len(stable_steady_states),len(stable_steady_states)), dtype = np.int16)
        attractor_counts = np.zeros(len(stable_steady_states), dtype = np.int16)

        y = list(zip(entry_times_tot,attractor_indices_tot))
        
        ## Counting number of times system enters each attractor
        for l in y:
            attractor_counts[l[1]] += 1

        ## Creating the attractor_transition_matrix
        j = 0
        while j < len(y)-1:
            attractor_transition_mat[(y[j])[1],(y[j+1])[1]] += 1
            j+=1

    return (entry_times_tot, attractor_indices_tot, attractor_transition_mat, attractor_counts)
    
           
#list(flatten([item[0] for item in switching[2]]))

#[item[0] if item.shape[0] == 1 else item.f for item in np.flatten(switching[2])]
#list(zip(*switching[2]))[0]

