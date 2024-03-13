import numpy as np
import MBP_reference_implementation as pot
from reference_implementation import *

kB    = 0.008314463 # kJ mol-1 K
mass  = 1           # amu

store_directory ='./simulation/ABOBA_300_1x/' 
trajectories    = 'run'

potential           = pot.MBLP()
potential_gradient  = potential.bias1X_gradient

## simulation 
position_inital     = np.array(np.random.rand(2)*[-0.5,1.5]) 
momentum_inital     = np.array([1,1])

n_trajs             = 5                   # runs
n_steps             = 2.5*10**8           # steps    
time_step           = 0.0005              # ps

friction            = 5.                  # ps-1
T                   = 300                 # K
eta                 = None
                   
batch_size          = 1000 
write_out_frequency = None
integration_scheme  = ABOBA


for t in range(n_trajs):
    #eta =  np.loadtxt("./eta/eta_"+str(t)+'.dat') # if shared random numbers
    const = Simulation(potential_gradient=potential_gradient, 
                       position_inital=position_inital, 
                       momentum_inital=momentum_inital,
                       n_steps=n_steps,                 
                       time_step=time_step,             
                       mass=mass,                      
                       friction=friction,                 
                       Boltzmann=kB,         
                       temperature=T,              
                       eta=None) #eta) 
    
    const.simulate_batches(integration_scheme,
                           batch_size,
                           write_out_frequency, 
                           #file_name=str(t), 
                           store_directory= store_directory + trajectories  +'_'+str(t)+'/')


