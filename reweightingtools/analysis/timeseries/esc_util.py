import numpy as np

def esc_condition(trajectory, esc_limit=0.45): 
    '''Escape condition for a dissociation potential.
    
    Arguments:
    trajectory (array: (N,)): time discretized trajectory give the x-position for each of the N timesteps
    esc_limit  (float)      : escape limit, depending on potential energy surface in positive direction
    
    Return:
    traj_pos_esc       (array: (N,)) : every point index in trajectory after a positive escape
    traj_prae_pos_esc  (array: (N,)) : every point index in trajectory bevor a positive escape 

    '''
    traj_pos_esc      = np.where(trajectory >=  esc_limit)[0] 
    traj_prae_pos_esc = np.where(trajectory <   esc_limit)[0]          
    return traj_pos_esc, traj_prae_pos_esc

def esc_condition_DW(trajectory, esc_limit=0.8):
    '''Escape condition for a double well potential.
    
    Arguments:
    trajectory (array: (N,)): time discretized trajectory give the x-position for each of the N timesteps
    esc_limit  (float)      : escape limit, depending on potential energy surface
    
    Return:
    traj_pos_esc       (array: (N,)) : every point index in trajectory after a positive escape
    traj_prae_pos_esc  (array: (N,)) : every point index in trajectory bevor a positive escape 

    traj_neg_esc       (array: (N,)) : every point index in trajectory after a negative escape
    traj_prae_neg_esc  (array: (N,)) : every point index in trajectory bevor a negative escape
    '''
    traj_pos_esc      = np.where(trajectory >=  esc_limit)[0] 
    traj_prae_pos_esc = np.where(trajectory <   esc_limit)[0]     
    
    traj_neg_esc      = np.where(trajectory <= -esc_limit)[0]  
    traj_prae_neg_esc = np.where(trajectory >  -esc_limit)[0]     

    if traj_neg_esc.size > traj_pos_esc.size:
        lpm_neg=True
    elif traj_neg_esc.size < traj_pos_esc.size:
        lpm_neg=False
    else:
        lpm_neg=None
    
    return traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc, lpm_neg

def list_esc_conditions(trajectory):
    '''Function divide long trajectories into sub-trajectoies and applies :func:`esc_condition_DW()` 
    on each sub-trajectory. 
    
    Arguments:
    trajectory        (array: (N,)) : time discretized trajectory give the x-position for each of the N timesteps
    esc_condition_DW  (function)    : give escape condition for a double well potential 
    
    Return:
    List     (list: N_sub,4) : give the four escape conditions (traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, 
    traj_prae_neg_esc) obtained from :func:`esc_condition_DW()` for each sub-trajectory (N_sub)
    lpm_neg  (boolean)       : give information about longer polulated minimum, True-> negative or False -> positive 
    '''
    List=[]
    List_lpm_neg=[]
    for i in range(trajectory.shape[0]):
        traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc, lpm_neg = esc_condition_DW(trajectory[i])
        List.append([traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc])
        List_lpm_neg.append(lpm_neg)
    lpm_neg=max(set(List_lpm_neg), key=List_lpm_neg.count) 
    return List, lpm_neg

def traj_prae_esc_p(traj_pos_esc, traj_prae_pos_esc):
    traj_prae_esc_to_pos      = traj_prae_pos_esc[np.where((traj_prae_pos_esc+1)==traj_pos_esc[:,None])[1]]
    return traj_prae_esc_to_pos 

def traj_prae_esc_(traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc):
    '''Function to give all states :math:`t` in the trajectory that cross a barrier after the next step :math:`t+\tau`.
    
    Arguments:
    traj_pos_esc       (array: (N,)) : every point index in trajectory after a positive escape
    traj_prae_pos_esc  (array: (N,)) : every point index in trajectory bevor a positive escape 

    traj_neg_esc       (array: (N,)) : every point index in trajectory after a negative escape
    traj_prae_neg_esc  (array: (N,)) : every point index in trajectory bevor a negative escape
    
    Return:
    traj_prae_esc_to_neg (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the negative barrier after the next step :math:`t+\tau`
    traj_prae_esc_to_pos (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the positive barrier after the next step :math:`t+\tau`
    '''
    traj_prae_esc_to_neg      = traj_prae_neg_esc[np.where((traj_prae_neg_esc+1)==traj_neg_esc[:,None])[1]] 
    traj_prae_esc_to_pos      = traj_prae_pos_esc[np.where((traj_prae_pos_esc+1)==traj_pos_esc[:,None])[1]]
    return traj_prae_esc_to_neg , traj_prae_esc_to_pos 

def traj_prae_esc_List(List_esc_conditions,trajectory,L):
    '''Function to give all states :math:`t` in the trajectory that cross a barrier after the next step :math:`t+\tau`
    for a list of trajectories.
    
    Arguments:
    List_esc_conditions  (list: N_sub,4) : give the four escape conditions (traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, 
    traj_prae_neg_esc) obtained from :func:`esc_condition_DW()` for each sub-trajectory (N_sub)
    trajectory           (array: (N,))   : time discretized trajectory give the x-position for each of the N timesteps
    L                    (int)           : Default=1000 -> can we improve?
    
    Return:
    Traj_prae_esc_to_neg (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the negative barrier after the next step :math:`t+\tau`
    Traj_prae_esc_to_pos (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the positive barrier after the next step :math:`t+\tau`
    '''
    List_traj_prae_esc_to_neg=[]
    List_traj_prae_esc_to_pos=[]
    for i in range(len(List_esc_conditions)):
        traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc=List_esc_conditions[i]
        traj_prae_esc_to_neg , traj_prae_esc_to_pos = traj_prae_esc_(traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc)
        List_traj_prae_esc_to_neg.append(traj_prae_esc_to_neg)
        List_traj_prae_esc_to_pos.append(traj_prae_esc_to_pos)
    
    Traj_prae_esc_to_neg=np.concatenate([List_traj_prae_esc_to_neg[i]+(i*L) for i in range(len(trajectory))])
    Traj_prae_esc_to_pos=np.concatenate([List_traj_prae_esc_to_pos[i]+(i*L) for i in range(len(trajectory))])

    return Traj_prae_esc_to_neg, Traj_prae_esc_to_pos

def find_jumps(Traj_prae_esc_more_trans,Traj_prae_esc_less_trans):
    '''Function to find all states :math:`t+\tau` in the trajectory that crossed a barrier.
    
    Arguments:
    Traj_prae_esc_to_neg (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the negative barrier after the next step :math:`t+\tau`
    Traj_prae_esc_to_pos (array: (N,)) : give all states :math:`t` in the trajectory that cross 
    the positive barrier after the next step :math:`t+\tau`              
    
    Return:
    jump_more (array: (N,)) : give all states :math:`t+\tau` in the trajectory that crossed 
    a barrier, give the direction to which more transitions are observerd
    jump_less (array: (N,)) : give all states :math:`t+\tau` in the trajectory that crossed 
    a barrier, give the direction to which more transitions are observerd
    '''
    phasen   = [(np.where((Traj_prae_esc_more_trans < i)!=False)[0].size) for i in Traj_prae_esc_less_trans ]
    prae_esc_more_trans, prae_esc_less_trans = np.unique(phasen, return_index=True)
    ## define the first point of trajectory in new state
    if prae_esc_more_trans[-1]==Traj_prae_esc_more_trans.size:
        prae_esc_more_trans=prae_esc_more_trans[:-1]
    jump_more = Traj_prae_esc_more_trans[prae_esc_more_trans]+1 
    jump_less = Traj_prae_esc_less_trans[prae_esc_less_trans]+1
    return jump_more, jump_less

def esc_times_(jump_to_neg,jump_to_pos,trajectory,lpm_neg,timestep): ## ATTENTION
    '''Function to determine escape rates and give the corresponding escape direction.
    :math:`t_esc = t_{i+1}-t_i` under the condition that {i+1} and i are in two different minima 
    :math:`esc_rate = 1/(1/N_esc \sum^{N_esc}_{j=0} t_esc(j)`
    
    Arguments:
    jump_to_neg (array: (N,)) : give all states :math:`t+\tau` in the trajectory that crossed 
    the negative barrier
    jump_to_pos (array: (N,)) : give all states :math:`t+\tau` in the trajectory that crossed 
    the positive barrier           
    
    Return:
    esc_rate_lpm_to_spm (float) : escape rate for transitions from longer to shorter populated minimum
    esc_rate_spm_to_lpm (float) : escape rate for transitions from shorter to longer populated minimum
    '''
    esc_times    = np.sort(np.concatenate((jump_to_neg,jump_to_pos))) 
    start_traj   = esc_times[0]
    first_minima = trajectory[esc_times[0]]
    
    esc_times = esc_times[1:]-esc_times[:-1]  
    esc_times_first_jump  = esc_times[::2]
    esc_times_second_jump = esc_times[1::2]
    
    if first_minima > 0.45: ## ATTENTION first_minima > 0.8
        print('The first minimum populated is in positive x-direction.')
        if lpm_neg:
            print('The trajectory does noch start in the longer populated minimum.')
            esc_times_spm_to_lpm = esc_times_first_jump
            esc_times_lpm_to_spm = esc_times_second_jump

        elif lpm_neg==False:
            print('The trajectory start in the longer populated minimum.')
            esc_times_lpm_to_spm = esc_times_first_jump
            esc_times_spm_to_lpm = esc_times_second_jump            
            
    elif first_minima < 0.35: ## ATTENTION first_minima < -0.8
        print('The first minimum populated is in negative x-direction.')
        if lpm_neg==False:
            print('The trajectory does noch start in the longer populated minimum.')
            esc_times_spm_to_lpm = esc_times_first_jump
            esc_times_lpm_to_spm = esc_times_second_jump

        elif lpm_neg:
            print('The trajectory start in the longer populated minimum.')
            esc_times_lpm_to_spm = esc_times_first_jump
            esc_times_spm_to_lpm = esc_times_second_jump 
        
    
    esc_times_lpm_to_spm = esc_times_lpm_to_spm 
    esc_rate_lpm_to_spm = 1/(esc_times_lpm_to_spm.mean()*timestep)
    print('The escape rate for transitions from longer to shorter populated minimum: ',esc_rate_lpm_to_spm)
    esc_times_spm_to_lpm = esc_times_spm_to_lpm 
    esc_rate_spm_to_lpm = 1/(esc_times_spm_to_lpm.mean()*timestep)
    print('The escape rate for transitions from shorter to longer populated minimum: ',esc_rate_spm_to_lpm)   
    return esc_rate_lpm_to_spm, esc_rate_spm_to_lpm
