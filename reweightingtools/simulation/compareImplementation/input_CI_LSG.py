'''This file provides input for test runs in various implementation methods.
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

from reweightingtools.simulation.openmmSimulation import openMM_biased_simulation, restraints_harmonic_force

from openmm.unit import nanometer, kelvin, picoseconds, femtoseconds, kilojoules_per_mole
from openmm import CustomIntegrator, CustomExternalForce, Platform
from openmm.app import PME, HBonds, GromacsGroFile, GromacsTopFile, simulation, CheckpointReporter
from openmmtools.constants import kB
from openmmplumed import PlumedForce
from reweightingtools.simulation.openmmSimulation.reweightingreporter import *

from openMM_CI_functions import global_variable_names, addComputeTemperatureDependentConstants, PositionReporter, VelocityReporter, RandomNumberReporter, BiasReporter, set_restraints, restraints_harmonic_force, DeltaEtaReporter, TotalForceReporter
#%% input 
nsteps=101
nstxout=1
T=300
xi=2
step=0.5
mass=1
nbC=0.49
sol=3*49
nParticles=1
mol=3*nParticles
solvent='HOH'
restraints_kwargs={
    'multiple_restraints' : True,
    'atom_name' : [('CA'),('I')] ,
    'force_constants' : [200000.0,500000.0],
    'restraint_x' : [True, True],
    'restraint_y' : [True, True],
    'restraint_z' : [False, True]
}  
restraints_func= [restraints_harmonic_force,restraints_harmonic_force]
restraints_kwargs=restraints_kwargs
forcefield='amber99.ff'
plumed_file='plumedCOLVAR.dat'
colvar_file='COLVAR'
externalForce_file='bias_5z.txt'
gro_input='ICaI_xs'
top_input='topol_xs'
nonbondedMethod=PME
constraints=HBonds
PlatformByNamen='CPU'
constraint_tolerance=1e-8
inputMD='./inputMD/'
plumedMD='./plumedMD/'
