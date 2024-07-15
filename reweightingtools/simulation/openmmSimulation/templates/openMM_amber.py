#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Sep  8 14:51:57 2023

@author: schaefej51
openmm simulation script for a unbiased simulation
"""
from openmm.unit import *
from openMM_wrapper import *
from simtk.openmm.app import *
from simtk.openmm import *

# define simulation input
equisteps= 50000
nsteps= 250000000
integrator_scheme= 'Langevin'
integrator_splitting='R V O V R'
nstxout=50


outputMD ='unbiasedMD/'
inputMD ='inputMD/'
plumedMD ='plumedMD/'
PlatformByNamen ='CPU'
inpcrd_input ='butane'
prmtop_input ='butane'

equilibration_integrator_scheme ='Langevin'
equilibration_integrator_splitting ='R V O V R'
temperature =300*kelvin
collision_rate =10*(picoseconds)**(-1)
timestep =2*femtoseconds
nonbondedMethod=PME
nonbondedCutoff=1*nanometer
constraints=HBonds
solvent ='HOH'
fix_system =False
dcd_output=True
pdp_output=False
data_output=False

# directory structure check
cwd=os.getcwd() 
input_directory=os.path.join(cwd,inputMD)
if os.path.isdir(input_directory):
    if not os.listdir(input_directory):
        print("MD input directory is empty.")
else:
    sys.exit("MD input directory doesn't exist")

output_directory=os.path.join(cwd,outputMD) 
if os.path.isdir(output_directory):
    pass
else:
    os.mkdir(output_directory)

bias_directory=os.path.join(cwd,plumedMD)
if os.path.isdir(bias_directory):
    if not os.listdir(bias_directory):
        print("Directory for PLUMED is empty.")
    pass
else:
    sys.exit("Directory for PLUMED input doesn't exist")

# load structure, topology file and define platform
inpcrd, prmtop, platform = get_inpcrdprmtopplatform(PlatformByNamen, # could rename
                                                    None,
                                                    input_directory,
                                                    output_directory,
                                                    inpcrd_input,
                                                    prmtop_input
                                                    )
         
# equilibration
# equilibration integrator
equilibration_integrator_argws=dict()   
equilibration_integrator_argws['integrator_scheme']=equilibration_integrator_scheme
equilibration_integrator_argws['temperature']=temperature
equilibration_integrator_argws['splitting']=equilibration_integrator_splitting
equilibration_integrator_argws['collision_rate']=collision_rate
equilibration_integrator_argws['timestep']=timestep
equilibration_integrator=set_integrator(**equilibration_integrator_argws)
equilibrationsystem = prmtop.createSystem(implicitSolvent=OBC2,   # top or prmtop file
                                          nonbondedMethod=CutoffNonPeriodic, 
                                          nonbondedCutoff=1.0, 
                                          constraints=None, 
                                          rigidWater=True)   

# constrain molecule for equilibration by setting masses to zero
if fix_system:
    masses = []
    for atom in prmtop.topology.atoms():  # top or prmtop file 
        if atom.residue.name != solvent:
            masses.append(equilibrationsystem.getParticleMass(atom.index))
            equilibrationsystem.setParticleMass(atom.index, 0*amu) 
# set dependecies for short equilibration MD
equilibration = simulation.Simulation(prmtop.topology,  # top or prmtop file 
                                      equilibrationsystem, 
                                      equilibration_integrator, 
                                      platform)
equilibration.context.setPositions(inpcrd.positions) # gro or inpcrd file
# Minimising solvent
print('*** Minimizing  ...')
equilibration.minimizeEnergy()
print('*** Equilibrating  ...')
equilibration.context.setVelocitiesToTemperature(temperature)
equilibration.step(equisteps) 
# get equilibrated positions
positions = equilibration.context.getState(getPositions=True).getPositions()

# simulation
log = open(output_directory + "/MD.log", 'a')
log.write('##               _____________________________________________________________________\n' )
log.write('## S T A R T E D \n' )                                  
log.write('##               _____________________________________________________________________\n' )
log.write('integrator_scheme:  ' + str(integrator_scheme) + " \n")
log.write('nsteps           :  ' + str(nsteps) + " \n")
log.write('nstxout          :  ' + str(nstxout) + " \n")
log.write('temperature      :  ' + str(temperature) + " \n")
log.write('collision_rate   :  ' + str(collision_rate) + " \n")
log.write('timestep         :  ' + str(timestep) + " \n")
log.write('splitting        :  ' + str(integrator_splitting) + " \n")
log.write('nonbondedMethod  :  ' + str(nonbondedMethod) + " \n")
log.write('nonbondedCutoff  :  ' + str(nonbondedCutoff) + " \n")
log.write('constraints      :  ' + str(constraints) + " \n")
log.write("Simulation Start " + str(dati.now()) + "\n" )
log.close();    

# integration
integrator_argws=dict()   
integrator_argws['integrator_scheme']=integrator_scheme
integrator_argws['temperature']=temperature
integrator_argws['splitting']=integrator_splitting
integrator_argws['nstxout']=nstxout
integrator_argws['collision_rate']=collision_rate
integrator_argws['timestep']=timestep
integrator=set_integrator(**integrator_argws)

# define system, simulator and give initial position
system = prmtop.createSystem(implicitSolvent=OBC2, # top or prmtop file 
                             nonbondedMethod=CutoffNonPeriodic, 
                             nonbondedCutoff=1.0, 
                             constraints=None, 
                             rigidWater=True)    

MDsimulation = simulation.Simulation(prmtop.topology, system, integrator, platform) # gro or inpcrd file
MDsimulation.context.setPositions(positions)
# output
if dcd_output:
    output_dcdTraj = "/trajectory.dcd" 
    MDsimulation.reporters.append(dcdreporter.DCDReporter(output_directory+output_dcdTraj, 
                                                          nstxout))
if pdp_output:
    output_pdbTraj = "/trajectory.pdb"  
    MDsimulation.reporters.append(pdbreporter.PDBReporter(output_directory+output_pdbTraj, 
                                                          nstxout))
if data_output:
    output_data    = "/data.txt"       
    MDsimulation.reporters.append(statedatareporter.StateDataReporter(output_directory+output_data, 
                                                                      nstxout,
                                                                      step=True,
                                                                      potentialEnergy=True,
                                                                      kineticEnergy=True,
                                                                      temperature=True, 
                                                                      separator=' '))
output_chckpt  = "/chckpt"                
MDsimulation.reporters.append(CheckpointReporter(output_directory+output_chckpt, 
                                                 nstxout))
print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
MDsimulation.step(nsteps)  
print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))

log = open(output_directory + "/MD.log", 'a')
log.write("Simulation End " + str(dati.now()) + "\n" )
log.close();