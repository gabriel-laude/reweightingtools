''' This is a template script to run python MD simulation with the pythonSimulation module.
'''
#%% import reweightingtools
import numpy as np
from datetime import datetime as dati
import os 
import sys
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = os.getcwd()
dir_path = os.path.join(dir_path, os.pardir) 
module_dir = os.path.abspath(dir_path)
sys.path.insert(0, module_dir)

from reweightingtools.integration.integrators import ABOBA
from reweightingtools.potential.MBP import pyMueller
from reweightingtools.simulation.pythonSimulation.simulation import Simulation 

#%% import reweightingtools
store_directory='./python_MBP/'
trajectories='run'
batch_size=None 
write_out_frequency=None
integration_scheme=ABOBA
n_trajs=5 # runs
nsteps=50_000_000
nstxout=1
T=300 # K
kB=0.008314 # kJ mol-1 K
xi=5 # ps-1
timestep=0.0005
nParticles=1
mass=1 # amu
startingPositions=(np.random.rand(nParticles,3)*np.array([1.,0.5,0.]))
startingVelocity=(np.zeros((nParticles,3)))

mb=pyMueller()
potential_class=mb
potential           = mb.potential
potential_gradient  = mb.gradient

## simulation 
for t in range(n_trajs):
    simulation = Simulation(potential_gradient=potential_gradient, 
                       position_inital=startingPositions[0][:2], 
                       momentum_inital=startingVelocity[0][:2],
                       n_steps=nsteps,                 
                       time_step=timestep, 
                       potential_class=potential_class,            
                       mass=mass,                      
                       friction=xi,                 
                       Boltzmann=kB,         
                       temperature=T,              
                       eta=None) 
    
    simulation.run(h5py_format=True, 
                   _integration_scheme=ABOBA, 
                   write_out_frequency=write_out_frequency, 
                   store_directory=store_directory + trajectories  +'_'+str(t)+'/', 
                   batch_size=batch_size)