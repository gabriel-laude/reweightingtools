import os
import sys

# Add the directory containing your module to the Python path (before importing it)
module_dir = os.path.abspath('/home/schaefej51/Documents/2_Projects/reweightingtools')
sys.path.insert(0, module_dir)

from reweightingtools.erscapeRates.esc_util import *
import numpy as np

def esc_rates_DW(trajectory,timestep):
    r"""Base function to determine escape rates in double well potentials. 

    SOFAR: 
      - only trajectories of shape :math:`10**n \forall n \in \mathbb{R}` 
      - only for DW potential -> could be expanded to triple well ?
      - only for one escape condition -> hard coded to match one potential with pos and neg minima   

    Args:
        trajectory (array: (N,)): time discretized trajectory give the x-position for each of the N timesteps
        timestep   (float)      : integration timestep used to obtaine the trajectory
    
    Return: tupel of direction dependend escape rates
        esc_rate_lpm_to_spm (float) : escape rate for transitions from longer to shorter populated minimum
        esc_rate_spm_to_lpm (float) : escape rate for transitions from shorter to longer populated minimum
    """
    
    ##     ToDo: Part to define different escape conditions
    ## -> different potentials possible ?
    ## -> esc_condition_DW mit conditions
    
    ## Prepare trajectory, devide trajectory according to the condition, 
    ## give all points in trajectory that jump in the next step over a barrier
    if trajectory.size > 999:
        print('Your trajectory is '+str(trajectory.size)+' steps long.')
        L    = 1000
        traj = trajectory.reshape((-1,L))
        print('For computational purpos we reshape the trajectory in '+str(traj.shape)+' sub-trajectories.')
        List_esc_conditions, lpm_neg               = list_esc_conditions(traj)
        Traj_prae_esc_to_neg, Traj_prae_esc_to_pos = traj_prae_esc_List(List_esc_conditions,traj,L)
        if lpm_neg:
            print('The negative minimum is the longer populated minimum.')
        else:
            print('The positive minimum is the longer populated minimum.')
    else:
        print('Your trajectory is '+str(trajectory.size)+' steps long.')
        traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc, lpm_neg = esc_condition_DW(trajectory, esc_limit=0.8)
        Traj_prae_esc_to_neg, Traj_prae_esc_to_pos = traj_prae_esc_(traj_pos_esc, traj_prae_pos_esc, traj_neg_esc, traj_prae_neg_esc)
        if lpm_neg:
            print('The negative minimum is the longer populated minimum.')
        else:
            print('The positive minimum is the longer populated minimum.')
            
    ## find in which direction more jumps take place
    if Traj_prae_esc_to_neg.size > Traj_prae_esc_to_pos.size:
        print('More transitions to the minimum at the negative x-direction are counted.')
        jump_to_neg, jump_to_pos = find_jumps(Traj_prae_esc_to_neg,Traj_prae_esc_to_pos)
    
    else:
        print('More transitions to the minimum at the positive x-direction are counted.')
        jump_to_pos, jump_to_neg = find_jumps(Traj_prae_esc_to_pos,Traj_prae_esc_to_neg)

    ## define direction dependend escape rates      
    esc_rate_lpm_to_spm, esc_rate_spm_to_lpm = esc_times_(jump_to_neg,jump_to_pos,trajectory,lpm_neg,timestep)
    if esc_rate_lpm_to_spm > esc_rate_spm_to_lpm:
        print('Attention: Trajrctoy seems to be unphysical.') 
        
    return esc_rate_lpm_to_spm, esc_rate_spm_to_lpm
