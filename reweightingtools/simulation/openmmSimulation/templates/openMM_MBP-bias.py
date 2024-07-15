"""
Created on Thu Oct 12 13:15:34 2023

@author: schaefej51
openmm simulation script for a biased simulation and reweighting and OpenMM Mueller Brown Potential custom force 
"""
import numpy as np

from reweightingtools.potential.MBP import mmMueller, LinearBias
from openmm.unit import *
from datetime import datetime as dati

from openmm import *
from openmm.app import * #ToDo: could be nicer

# define simulation input
output_directory =  './'

biased= True
equilibration= True  

equisteps= 1000000 
nsteps= 500000000 
nstxout=1

integrator_scheme='ABOBA'

temperature=300*kelvin
kB=0.008314
collision_rate=5*(picoseconds)**(-1)
timestep=0.5*femtoseconds
nParticles=1
mass=1

platform='CPU'
constraint_tolerance=1e-8

startingPositions= (np.random.rand(nParticles, 3) * np.array([-0.5,1.5,1])) 

analyticforce = mmMueller()
if biased:
    perturbation = LinearBias()
else:
    perturbation = None

# logging    
log = open(output_directory + "MD.log", 'a')
log.write('##               _____________________________________________________________________\n' )
log.write('## S T A R T E D \n' )                                  
log.write('##               _____________________________________________________________________\n' )
log.write('integrator_scheme:  ' + str(integrator_scheme) + " \n")
log.write('nsteps          :  ' + str(nsteps) + " \n")
log.write('nstxout          :  ' + str(nstxout) + " \n")
log.write('temperature      :  ' + str(temperature) + " \n")
log.write('collision_rate   :  ' + str(collision_rate) + " \n")
log.write('timestep         :  ' + str(timestep) + " \n")
log.write("Simulation Start " + str(dati.now()) + "\n" )
log.close();    

# integrator constants
gamma=collision_rate 
dt=timestep
h=timestep/1

# initialize custom integrator like CI -> ThermostatedIntegrator -> LangevinIntegrator -> LangevinSplittingGirsanov
Integrator = CustomIntegrator(dt)
def global_variable_names(Int):
    """The set of global variable names defined for this integrator."""
    return set([ Int.getGlobalVariableName(index) for index in range(Int.getNumGlobalVariables()) ])
def addComputeTemperatureDependentConstants(Int, compute_per_dof):
    # First check if flag variable already exist.
    if not 'has_kT_changed' in global_variable_names(Int):
        Int.addGlobalVariable('has_kT_changed', 1)

    # Create if-block that conditionally update the per-DOF variables.
    Int.beginIfBlock('has_kT_changed = 1')
    for variable, expression in compute_per_dof.items():
        Int.addComputePerDof(variable, expression)
    Int.addComputeGlobal('has_kT_changed', '0')
    Int.endBlock()

# Velocity mixing parameter: current velocity component
Integrator.addGlobalVariable("a", np.exp(-gamma * h))
# Velocity mixing parameter: random velocity component
Integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * gamma * h)))
# Positions before application of position constraints
Integrator.addPerDofVariable("x1", 0)
# Set constraint tolerance
Integrator.setConstraintTolerance(constraint_tolerance)

# Add global variables
Integrator.addPerDofVariable("sigma", 0)
Integrator.addGlobalVariable("n", 0)
Integrator.addGlobalVariable("timestep", dt)
## Add a variable for \tau the length of a path \omega; 
## here given by the write-out freuquency nstxout
Integrator.addGlobalVariable("tau", nstxout)      
## Add variables to enable sumation over the path 
## cf. J. Chem. Phys. 146, 244112 (2017) EQ:(29)
Integrator.addGlobalVariable("ndivtau", 0)
Integrator.addGlobalVariable("onedelta", 0)     
## Abb variable give the sum over the path                     
## cf. J. Chem. Phys. 146, 244112 (2017) EQ:(25) 
Integrator.addGlobalVariable("SumOverPath", 0)
Integrator.addGlobalVariable("M", 0)   
## Add variable for \eta and \Delta\eta needed to give reweighting factor M(\eta)
## cf. J. Chem. Phys. 154, 094102 (2021) EQ:(10)
Integrator.addPerDofVariable("Eta0",0)
Integrator.addPerDofVariable("DeltaEta0",0)
Integrator.addPerDofVariable("ff0",0)

Integrator.addUpdateContextState()
Integrator.addGlobalVariable('kT', kB * temperature) 
addComputeTemperatureDependentConstants(Integrator, {"sigma": "sqrt(kT/m)"}) 
# Add random number
Integrator.addComputePerDof("Eta0","gaussian")

# "R V O V R" : (1, ['1/(b*sigma*m) * (1 + a) * timestep/2 * ff0'], "R U V O V R")
# update positions (and velocities, if there are constraints)
Integrator.addComputePerDof("x", "x + ((dt / 2) * v)")
Integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
Integrator.addConstrainPositions()  # x is now constrained
Integrator.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
Integrator.addConstrainVelocities()

# _get_delta_eta(self, idx):
 # Update forces
Integrator.addComputePerDof("ff0","f1")  
# Set delta eta 
Integrator.addComputePerDof("DeltaEta0", '1/(b*sigma*m) * (1 + a) * timestep/2 * ff0')  
        
# update velocities
Integrator.addComputePerDof("v", "v + ((dt / 2) * f / m)")
Integrator.addConstrainVelocities()

# _add_O_step(self, eta_idx): -> update velocities with stored eta
Integrator.addComputePerDof("v", "(a * v) + (b * sigma * Eta0)")
Integrator.addConstrainVelocities()

# update velocities
Integrator.addComputePerDof("v", "v + ((dt / 2) * f / m)")
Integrator.addConstrainVelocities()

# update positions (and velocities, if there are constraints)
Integrator.addComputePerDof("x", "x + ((dt / 2) * v)")
Integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
Integrator.addConstrainPositions()  # x is now constrained
Integrator.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
Integrator.addConstrainVelocities()

# Trick to enable sumation over the path 
# for n=0 and after tau steps of a path delta gives 0
# so the integrals for the new path are recalculate  
Integrator.addComputeGlobal("ndivtau", "n/tau")
Integrator.addComputeGlobal("onedelta","1 - delta(ndivtau-floor(ndivtau))") 
# Random number based reweighting factor logM(\eta)
# Sum over the path (SOP) of length \tau 
SOP = str()
SOP+="Eta0 * DeltaEta0 + 0.5 * (DeltaEta0 * DeltaEta0)"
Integrator.addComputeSum("SumOverPath", SOP) 
Integrator.addComputeGlobal('M', "M * onedelta + SumOverPath")
# Increase timestep n for the next integration step
Integrator.addComputeGlobal("n", "n + 1")

integrator=Integrator
   
#define system with analytic force and give initial position
system = System()
for i in range(nParticles):
    system.addParticle(mass)
    analyticforce.addParticle(i, [])
    if biased:
        perturbation.addParticle(i, [])
system.addForce(analyticforce)
if biased:
    perturbation.setForceGroup(1)
    system.addForce(perturbation)

context = Context(system, integrator)
context.setPositions(startingPositions)
context.setVelocitiesToTemperature(temperature)

if equilibration:
    for i in range(equisteps):
        state  = context.getState(getEnergy = True,
                                  groups = 0b00000000000000000000000000000000) 
        integrator.step(1)

# write out simulation data 
trajlength=int(nsteps/nstxout)
for i in range(trajlength):
    position = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    log = open(output_directory + "positions.txt", 'a')
    log.write(str(position[0][0]) +' '+str(position[0][1])+' ' +str(position[0][2]) + " \n")
    log.close();    
    
    state  = context.getState( getEnergy = True,
                              groups = 0b00000000000000000000000000000000) 
    Ekin = state.getKineticEnergy().value_in_unit(unit.kilojoules_per_mole)
    Temp=2*Ekin/(3*6.02214076e23*1.380649e-23 *0.001)
    log = open(output_directory + "temp.txt", 'a')
    log.write(str(Temp) + " \n")
    log.close();  
    etavalues=integrator.getPerDofVariableByName("Eta0")
    log = open(output_directory + "eta.txt", 'a')
    log.write(str(etavalues) + " \n")
    log.close(); 
        
    if biased:
        Mvalues=integrator.getGlobalVariableByName("M")
        state  = context.getState(getPositions = False, 
                                  getVelocities = False,
                                  getForces = False, getEnergy = True,
                                  getParameters = False, 
                                  enforcePeriodicBox = False,
                                  groups = 0b00000000000000000000000000000010) 
        gvalues = state.getPotentialEnergy().value_in_unit(unit.kilojoules_per_mole)
        log = open(output_directory + "ReweightingFactors.txt", 'a')
        log.write(str(Mvalues) + ' '+ str(gvalues)+ " \n")
        log.close(); 
    integrator.step(nstxout)
           

   
