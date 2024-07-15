#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:38:58 2023

@author: schaefej51

This is a collection of wrapper functions for openMM tasks.
"""
from openmm import *
from openmm.app import * #ToDo: could be nicer
try:
    from reweightingreporter import *   
    from _utils import *
except:
    from .reweightingreporter import *
    from ._utils import *
    
from openmmtools.integrators import LangevinIntegrator, LangevinSplittingGirsanov
from openmmplumed import PlumedForce 

from simtk.unit import nanometer, amu, kilojoules_per_mole # kelvin, picosecond, femtosecond, dalton

def set_integrator(**kwargs):
    ''' This is a low-level function to give the correct input scheme for the 
    integrator chosen.'''
    if kwargs['integrator_scheme'] == 'Langevin':
        integrator = LangevinIntegrator(temperature=kwargs['temperature'], 
                                        collision_rate=kwargs['collision_rate'], 
                                        timestep=kwargs['timestep'], 
                                        splitting=kwargs['splitting']
                                        )
    elif kwargs['integrator_scheme'] == 'LangevinWithGirsanov':
        integrator = LangevinSplittingGirsanov(nstxout=kwargs['nstxout'], 
                                               temperature=kwargs['temperature'], 
                                               collision_rate=kwargs['collision_rate'], 
                                               timestep=kwargs['timestep'], 
                                               splitting=kwargs['splitting']
                                               )
    return integrator

def set_restraints(func, system, top, gro, **kwargs):
    if func==None:
        pass
    elif 'multiple_restraints' in list(kwargs.keys()):
        print(kwargs['atom_name'])
        for restraints in range(len(kwargs['atom_name'])):
            rst_kwargs=dict()
            for key, value in kwargs.items():
                print(key, value )
                if key != 'multiple_restraints':
                    rst_kwargs[key]= value[restraints]
            print(rst_kwargs)
            func[restraints](system, top, gro, **rst_kwargs)   
    else:
        print('else')
        return func(system, top, gro, **kwargs)

def restraints_harmonic_force(system, top, gro, **kwargs):
    ''' Harmonic force binds each particle to its initial position, by adding a CustomExternalForce that
    set the energy of each particle equals a force_constants multiplied by the square of the periodic 
    distance between the particle’s current position (x, y, z) and a reference position (x0, y0, z0).
    Args:
        system: the OpenMM System object to simulate (or the name of an XML file with a serialized 
        System)
        top: topology stores the topological information about a system.
        gro: constructs a set of atom positions from it, also contains some topological information, such 
        as elements and residue names
        atom_name: tuple of strings refering the topology abbreviation of the atom which should 
        be restrained
        force_constants: float, defines the strength of the replacement force
        restraint_x: (or _y, _z) bool defining if a harmonic force is defined in this direction
    Ref: 
        https://openmm.github.io/openmm-cookbook/dev/notebooks/restraints_constraints_forces/Restraining%20Atom%20Positions.html '''
    
    atom_name=kwargs['atom_name']
    force_constants=kwargs['force_constants']
    restraint_x=kwargs['restraint_x']
    restraint_y=kwargs['restraint_y']
    restraint_z=kwargs['restraint_z']
    
    if (restraint_x, restraint_y, restraint_z) == (True, True, True):
        restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        
        restraint.addGlobalParameter('k', force_constants*kilojoules_per_mole/nanometer)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')
        
        restraint.setForceGroup(3)  #ATTENTION
        system.addForce(restraint)

        for atom in top.topology.atoms():
            if atom.name in atom_name:
                print(atom.name)
                restraint.addParticle(atom.index, gro.positions[atom.index])
                
    elif (restraint_x, restraint_y, restraint_z) == (True, True, False):
        restraint_xy = CustomExternalForce('k_xy*periodicdistance(x, y, z0, x0, y0, z0)^2')
        
        restraint_xy.addGlobalParameter('k_xy', force_constants*kilojoules_per_mole/nanometer)
        restraint_xy.addPerParticleParameter('x0')
        restraint_xy.addPerParticleParameter('y0')
        restraint_xy.addPerParticleParameter('z0')
        
        restraint_xy.setForceGroup(4)  #ATTENTION
        system.addForce(restraint_xy)

        for atom in top.topology.atoms():
            if atom.name in atom_name:
                print(atom.name)
                restraint_xy.addParticle(atom.index, gro.positions[atom.index])
    
def get_grotopplatform(PlatformByNamen,
                       forcefield_directory,
                       input_directory,
                       output_directory,
                       gro_input,
                       top_input
                      ):
    '''This is a low-level function to provide the structure .gro and the topology .top file,
    as well as the platform for the openMM simulation.'''
    # logging
    log = open(output_directory + "/MD.log", 'a')
    log.write('## O P E N M M  S I M U L A T I O N \n' )                                
    log.write('## created '+ str(dati.now()) + "\n" )
    log.write('## ____________________________________________________________________________________\n' )
    log.write('forcefield_directory:  ' + str(forcefield_directory) + " \n")
    log.write('input_directory     :  ' + str(input_directory) + " \n")
    log.write('top_input           :  ' + str(top_input) + " \n")
    log.write('gro_input           :  ' + str(gro_input) + " \n")
    log.close();
    # Load .gro and .top files 
    gro = GromacsGroFile(input_directory + str(gro_input) + '.gro')
    top = GromacsTopFile(input_directory + str(top_input) + '.top',
                         periodicBoxVectors=gro.getPeriodicBoxVectors(),
                         includeDir=forcefield_directory
                        )
    # Defining Simulation Dependencies 
    platform = Platform.getPlatformByName(PlatformByNamen)
    return gro, top, platform

def get_inpcrdprmtopplatform(PlatformByNamen, # could rename
                       forcefield_directory,
                       input_directory,
                       output_directory,
                       inpcrd_input,
                       prmtop_input
                      ):
    '''This is a low-level function to provide the structure .inpcrd and the topology .prmtop file,
    as well as the platform for the openMM simulation.'''
    # logging
    log = open(output_directory + "/MD.log", 'a')
    log.write('## O P E N M M  S I M U L A T I O N \n' )                                
    log.write('## created '+ str(dati.now()) + "\n" )
    log.write('## ____________________________________________________________________________________\n' )
    log.write('forcefield_directory:  ' + str(forcefield_directory) + " \n")
    log.write('input_directory     :  ' + str(input_directory) + " \n")
    log.write('prmtop_input        :  ' + str(prmtop_input) + " \n")
    log.write('inpcrd_input        :  ' + str(inpcrd_input) + " \n")
    log.close();
    # Load .gro and .top files 
    gro = AmberInpcrdFile(input_directory + str(inpcrd_input) + '.inpcrd')
    top = AmberPrmtopFile(input_directory + str(prmtop_input) + '.prmtop')
    # Defining Simulation Dependencies 
    platform = Platform.getPlatformByName(PlatformByNamen)
    return gro, top, platform

def make_equilibration(top, 
                       gro, 
                       platform, 
                       temperature, 
                       equisteps,
                       equilibration_integrator_scheme,
                       equilibration_integrator_splitting,
                       collision_rate,
                       timestep,
                       nonbondedMethod,
                       nonbondedCutoff,
                       constraints,
                       solvent,
                       fix_system
                      ):
    '''This is a low-level function to equilibrate the openMM simulation system'''
    # equilibration integrator
    equilibration_integrator_argws=dict()   
    equilibration_integrator_argws['integrator_scheme']=equilibration_integrator_scheme
    equilibration_integrator_argws['temperature']=temperature
    equilibration_integrator_argws['splitting']=equilibration_integrator_splitting
    equilibration_integrator_argws['collision_rate']=collision_rate
    equilibration_integrator_argws['timestep']=timestep
    equilibration_integrator=set_integrator(**equilibration_integrator_argws)
    equilibrationsystem=top.createSystem(nonbondedMethod=nonbondedMethod, 
                                         nonbondedCutoff=nonbondedCutoff, 
                                         constraints=constraints)
    # constrain molecule for equilibration by setting masses to zero
    if fix_system:
        masses = []
        for atom in top.topology.atoms():
            print(atom.residue.name)
            if atom.residue.name != solvent:
                print(atom.residue.name)
                masses.append(equilibrationsystem.getParticleMass(atom.index))
                equilibrationsystem.setParticleMass(atom.index, 0*amu) 
    # set dependecies for short equilibration MD
    equilibration = simulation.Simulation(top.topology, 
                                          equilibrationsystem, 
                                          equilibration_integrator, 
                                          platform)
    equilibration.context.setPositions(gro.positions)
    # minimising solvent
    print('*** Minimizing  ...')
    equilibration.minimizeEnergy()
    print('*** Equilibrating  ...')
    equilibration.context.setVelocitiesToTemperature(temperature)
    equilibration.step(equisteps) 
    # get equilibrated positions
    positions = equilibration.context.getState(getPositions=True).getPositions()
    return positions

def make_simulation(positions, 
                    top, 
                    gro, 
                    plumedCommittor,
                    platform,
                    nstxout, 
                    nsteps,
                    integrator_scheme,
                    temperature,
                    collision_rate, 
                    timestep, 
                    integrator_splitting,
                    nonbondedMethod,
                    nonbondedCutoff,
                    constraints,
                    restraints_func, 
                    restraints_kwargs,
                    output_directory,
                    restart=False,
                    dcd_output=False,
                    pdp_output=False,
                    data_output=False
                   ):
    '''' This is a low-level wrapper function to performe a unbiased simulation using
    openMM and PLUMED..'''
    # logging
    logging_MDparameters(output_directory,
                         integrator_scheme,
                         nstxout,
                         temperature,
                         collision_rate,
                         timestep,
                         integrator_splitting,
                         nonbondedMethod,
                         nonbondedCutoff,
                         constraints,
                         plumedCommittor,
                         externalForce=None,
                         restart=restart
                         )
    # integration
    integrator_argws=dict()   
    integrator_argws['integrator_scheme']=integrator_scheme
    integrator_argws['temperature']=temperature
    integrator_argws['splitting']=integrator_splitting
    integrator_argws['nstxout']=nstxout
    integrator_argws['collision_rate']=collision_rate
    integrator_argws['timestep']=timestep
    integrator=set_integrator(**integrator_argws)
    #define system, simulator and give initial position
    system   = top.createSystem(nonbondedMethod=nonbondedMethod, 
                                nonbondedCutoff=nonbondedCutoff, 
                                constraints=constraints)  # maybe to include more constraints  
    system.addForce(PlumedForce(plumedCommittor.rstrip())) # set the committer function via plumed
    
    if restraints_func==None:
        pass
    else:
        set_restraints(restraints_func, system, top, gro, **restraints_kwargs)
    
    MDsimulation = simulation.Simulation(top.topology, system, integrator, platform)

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
    if restart:
        # give the last position of the system
        MDsimulation.loadCheckpoint(output_directory + output_chckpt)
    else:
        # give position of the system
        MDsimulation.context.setPositions(positions)
        
    print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
    MDsimulation.step(nsteps)  
    print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))
    
    log = open(output_directory + "/MD.log", 'a')
    log.write("Simulation End " + str(dati.now()) + "\n" )
    log.close();
    
def make_biased_simulation(positions, 
                           top, 
                           gro, 
                           plumedForce,
                           platform, 
                           nstxout, 
                           nsteps,
                           integrator_scheme,
                           temperature,
                           collision_rate, 
                           timestep, 
                           integrator_splitting,
                           nonbondedMethod,
                           nonbondedCutoff,
                           constraints,
                           output_directory,
                           restart=False,
                           dcd_output=False,
                           pdp_output=False,
                           data_output=False
                         ):
    ''' This is a low-level wrapper function to performe a biased simulation using 
    openMM and PLUMED.'''
    # logging
    logging_MDparameters(output_directory,
                         integrator_scheme,
                         nsteps,
                         nstxout,
                         temperature,
                         collision_rate,
                         timestep,
                         integrator_splitting,
                         nonbondedMethod,
                         nonbondedCutoff,
                         constraints,
                         plumedForce,
                         externalForce=None,
                         restart=restart
                         )
    # integration
    integrator_argws=dict()   
    integrator_argws['integrator_scheme']=integrator_scheme
    integrator_argws['temperature']=temperature
    integrator_argws['splitting']=integrator_splitting
    integrator_argws['nstxout']=nstxout
    integrator_argws['collision_rate']=collision_rate
    integrator_argws['timestep']=timestep
    integrator=set_integrator(**integrator_argws)
    #define system
    system=top.createSystem(nonbondedMethod=nonbondedMethod, 
                            nonbondedCutoff=nonbondedCutoff, 
                            constraints=constraints)  # possibility to include more constraints   
    # Setting bias via PLUMED force 
    perturbation = PlumedForce(plumedForce.rstrip())
    perturbation.setForceGroup(1)
    system.addForce(perturbation)
    # set dependencies for MD simulation
    MDsimulation = simulation.Simulation(top.topology, system, integrator, platform)
   
    # define output 
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
        
    output_rwgt    = "/ReweightingFactors.txt"  
    output_chckpt  = "/chckpt"                  
    MDsimulation.reporters.append(ReweightingReporter(output_directory+output_rwgt, 
                                                      nstxout, 
                                                      integrator, 
                                                      firtsPertubation=True, 
                                                      separator=' '))
    MDsimulation.reporters.append(CheckpointReporter(output_directory+output_chckpt, 
                                                     nstxout))
    
    if restart:
        # give the last position of the system
        MDsimulation.loadCheckpoint(output_directory + output_chckpt)
    else:
        # give position of the system
        MDsimulation.context.setPositions(positions)
        
    print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
    MDsimulation.step(nsteps)  
    print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))
    
    log = open(output_directory + "/MD.log", 'a')
    log.write("Simulation End " + str(dati.now()) + "\n" )
    log.close();
    
def make_externalForce_simulation(positions, 
                                  top, 
                                  gro, 
                                  plumedForce,
                                  externalForce,
                                  platform, 
                                  nstxout, 
                                  nsteps,
                                  integrator_scheme,
                                  temperature,
                                  collision_rate, 
                                  timestep, 
                                  integrator_splitting,
                                  nonbondedMethod,
                                  nonbondedCutoff,
                                  constraints,
                                  restraints_func, 
                                  restraints_kwargs,        
                                  output_directory,
                                  restart=False,
                                  dcd_output=False,
                                  pdp_output=False,
                                  data_output=False
                                 ):
    ''' This is a low-level wrapper function to performe a biased simulation using 
    openMM and PLUMED.'''
    # logging
    logging_MDparameters(output_directory,
                         integrator_scheme,
                         nsteps,
                         nstxout,
                         temperature,
                         collision_rate,
                         timestep,
                         integrator_splitting,
                         nonbondedMethod,
                         nonbondedCutoff,
                         constraints,
                         plumedForce,
                         externalForce
                         )
    # integration
    integrator_argws=dict()   
    integrator_argws['integrator_scheme']=integrator_scheme
    integrator_argws['temperature']=temperature
    integrator_argws['splitting']=integrator_splitting
    integrator_argws['nstxout']=nstxout
    integrator_argws['collision_rate']=collision_rate
    integrator_argws['timestep']=timestep
    integrator=set_integrator(**integrator_argws)
    #define system
    system=top.createSystem(nonbondedMethod=nonbondedMethod, 
                            nonbondedCutoff=nonbondedCutoff, 
                            constraints=constraints)  # possibility to include more constraints   
    nParticles = system.getNumParticles()
    print(nParticles)
    
    # set the bias via python
    ExternalForce = open(externalForce, 'r')
    ExternalForce = ExternalForce.read()
    perturbation = CustomExternalForce(ExternalForce)
        
    perturbation.addPerParticleParameter('x0')
    perturbation.addPerParticleParameter('y0')
    perturbation.addPerParticleParameter('z0')
        
    perturbation.setForceGroup(1)  #ATTENTION
    system.addForce(perturbation)

    for atom in top.topology.atoms():
        if atom.name in ('CA'):  #### ATTENTION: hard coded
            print(atom.name)
            perturbation.addParticle(atom.index, gro.positions[atom.index])
                
    # Setting COLVAR output via PLUMED force 
    perturbation2 = PlumedForce(plumedForce.rstrip())
    perturbation2.setForceGroup(2)
    system.addForce(perturbation2)
    
    if restraints_func==None:
        pass
    else:
        set_restraints(restraints_func, system, top, gro, **restraints_kwargs)
        
    # set dependencies for MD simulation
    MDsimulation = simulation.Simulation(top.topology, system, integrator, platform)

    # define output 
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
        
    output_rwgt    = "/ReweightingFactors.txt"  
    output_chckpt  = "/chckpt"                  
    MDsimulation.reporters.append(ReweightingReporter(output_directory+output_rwgt, 
                                                      nstxout, 
                                                      integrator, 
                                                      firtsPertubation=True, 
                                                      separator=' '))
    MDsimulation.reporters.append(CheckpointReporter(output_directory+output_chckpt, 
                                                     nstxout))
    if restart:
        # give the last position of the system
        MDsimulation.loadCheckpoint(output_directory + output_chckpt)
    else:
        # give position of the system
        MDsimulation.context.setPositions(positions)
        
    print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
    MDsimulation.step(nsteps)  
    print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))
    
    log = open(output_directory + "/MD.log", 'a')
    log.write("Simulation End " + str(dati.now()) + "\n" )
    log.close();
    
def make_externalForce_simulation_ETA(startingPositions,startingVelocities, 
                          top, 
                          gro, 
                          plumedForce,
                          externalForce,
                          platform, 
                          nstxout, 
                          nsteps,
                          integrator_scheme,
                          temperature,
                          collision_rate, 
                          timestep, 
                          integrator_splitting,
                          nonbondedMethod,
                          nonbondedCutoff,
                          constraints,
                          restraints_func, 
                          restraints_kwargs,        
                          output_directory,
                          restart=False,
                          dcd_output=False,
                          pdp_output=False,
                          data_output=False
                         ):
    ''' This is a low-level wrapper function to performe a biased simulation using 
    openMM and PLUMED.'''
    # logging
    logging_MDparameters(output_directory,
                         integrator_scheme,
                         nsteps,
                         nstxout,
                         temperature,
                         collision_rate,
                         timestep,
                         integrator_splitting,
                         nonbondedMethod,
                         nonbondedCutoff,
                         constraints,
                         plumedForce,
                         externalForce
                         )
    # integration
    integrator_argws=dict()   
    integrator_argws['integrator_scheme']=integrator_scheme
    integrator_argws['temperature']=temperature
    integrator_argws['splitting']=integrator_splitting
    integrator_argws['nstxout']=nstxout
    integrator_argws['collision_rate']=collision_rate
    integrator_argws['timestep']=timestep
    integrator=set_integrator(**integrator_argws)
    
    class PositionReporter(object):
        def __init__(self, file, reportInterval):
            self._out = open(file, 'w')
            self._reportInterval = reportInterval

        def __del__(self):
            self._out.close()

        def describeNextReport(self, simulation):
            steps = self._reportInterval - simulation.currentStep%self._reportInterval
            return (steps, False, False, True, False)

        def report(self, simulation, state):
            position = MDsimulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
            for i in range(nParticles):
                print(str(position[i][0]) +' '+str(position[i][1])+' ' +str(position[i][2]), file = self._out)

    class VelocityReporter(object):
        def __init__(self, file, reportInterval):
            self._out = open(file, 'w')
            self._reportInterval = reportInterval

        def __del__(self):
            self._out.close()

        def describeNextReport(self, simulation):
            steps = self._reportInterval - simulation.currentStep%self._reportInterval
            return (steps, False, False, True, False)

        def report(self, simulation, state):
            velocity = MDsimulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
            for i in range(nParticles):
                print(str(velocity._value[i][0]) +' '+str(velocity._value[i][1])+' ' +str(velocity._value[i][2]), file = self._out)


    class RandomNumberReporter(object):
        '''
        reporter for ML for momentum space
        '''
        def __init__(self, file, reportInterval):
            self._out = open(file, 'w')
            self._reportInterval = reportInterval

        def __del__(self):
            self._out.close()

        def describeNextReport(self, simulation):
            steps = self._reportInterval - simulation.currentStep%self._reportInterval
            return (steps, False, False, True, False)

        def report(self, simulation, state):
            etavalues=integrator.getPerDofVariableByName("Eta0")
            for i in range(nParticles):
                print(str(etavalues[i][0]) +' '+str(etavalues[i][1])+' ' +str(etavalues[i][2]), file = self._out)

    class BiasReporter(object):
        '''
        reporter for ML for momentum space
        '''
        def __init__(self, file, reportInterval):
            self._out = open(file, 'w')
            self._reportInterval = reportInterval

        def __del__(self):
            self._out.close()

        def describeNextReport(self, simulation):
            steps = self._reportInterval - simulation.currentStep%self._reportInterval
            return (steps, False, False, True, False)

        def report(self, simulation, state):
            bias=integrator.getPerDofVariableByName("ff0")
            for i in range(nParticles):
                print(str(bias[i][0]) +' '+str(bias[i][1])+' ' +str(bias[i][2]), file = self._out)


    #define system
    system=top.createSystem(nonbondedMethod=nonbondedMethod, 
                            nonbondedCutoff=nonbondedCutoff, 
                            constraints=constraints)  # possibility to include more constraints   
    nParticles = system.getNumParticles()
    print(nParticles)
    
    # set the bias via python
    ExternalForce = open(externalForce, 'r')
    ExternalForce = ExternalForce.read()
    perturbation = CustomExternalForce(ExternalForce)
        
    perturbation.addPerParticleParameter('x0')
    perturbation.addPerParticleParameter('y0')
    perturbation.addPerParticleParameter('z0')
        
    perturbation.setForceGroup(1)  #ATTENTION
    system.addForce(perturbation)

    for atom in top.topology.atoms():
        if atom.name in ('CA'):  #### ATTENTION: hard coded
            print(atom.name)
            perturbation.addParticle(atom.index, gro.positions[atom.index])
                
    # Setting COLVAR output via PLUMED force 
    perturbation2 = PlumedForce(plumedForce.rstrip())
    perturbation2.setForceGroup(2)
    system.addForce(perturbation2)
    
    if restraints_func==None:
        pass
    else:
        set_restraints(restraints_func, system, top, gro, **restraints_kwargs)
        
    # set dependencies for MD simulation
    MDsimulation = simulation.Simulation(top.topology, system, integrator, platform)

    # define output 
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
        
    output_rwgt    = "/ReweightingFactors.txt"  
    output_chckpt  = "/chckpt"                  
    MDsimulation.reporters.append(ReweightingReporter(output_directory+output_rwgt, 
                                                      nstxout, 
                                                      integrator, 
                                                      firtsPertubation=True, 
                                                      separator=' '))
    MDsimulation.reporters.append(CheckpointReporter(output_directory+output_chckpt, 
                                                     nstxout))
    MDsimulation.reporters.append(BiasReporter(output_directory + "/bias_force.txt",nstxout))
    MDsimulation.reporters.append(PositionReporter(output_directory + "/positions.txt",nstxout))
    MDsimulation.reporters.append(VelocityReporter(output_directory + "/velocities.txt",nstxout))   
    MDsimulation.reporters.append(RandomNumberReporter(output_directory + "/eta.txt",nstxout))


    if restart:
        # give the last position of the system
        MDsimulation.loadCheckpoint(output_directory + output_chckpt)
    else:
        # give position of the system
        MDsimulation.context.setPositions(startingPositions)
        startingPositions=MDsimulation.context.getState(getPositions=True).getPositions(asNumpy=True)
        np.save(output_directory+'/startingPositions',startingPositions)
        startingVelocities=MDsimulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
        np.save(output_directory+'/startingVelocities',startingVelocities)
        
    print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
    MDsimulation.step(nsteps)  
    print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))
    
    log = open(output_directory + "/MD.log", 'a')
    log.write("Simulation End " + str(dati.now()) + "\n" )
    log.close();
