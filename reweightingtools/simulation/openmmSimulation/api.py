#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jul 12 11:51:31 2023

@author: schaefej51
"""
# I M P O R T S
from openmm.unit import *
from ..openMM_wrapper import *

def openMM_simulation(forcefield:str,
                      equisteps:int,
                      nsteps:int,
                      integrator_scheme:str,
                      integrator_splitting:str,
                      nstxout:int,
                      outputMD:str='unbiasedMD/',
                      inputMD:str='inputMD/',
                      plumedMD:str='plumedMD/',
                      PlatformByNamen:str='CPU',
                      gro_input:str='box_ions',
                      top_input:str='topol',
                      plumed_file:str='plumedCommittor.dat',
                      colvar_file:str='COLVAR',
                      equilibration_integrator_scheme:str='Langevin',
                      equilibration_integrator_splitting:str='R V O V R',
                      temperature:float=300*kelvin, 
                      collision_rate:float=2*(picoseconds)**(-1),
                      timestep:float=1*femtoseconds,
                      nonbondedMethod=PME,
                      nonbondedCutoff=0.6*nanometer,
                      constraints=HBonds,
                      restraints_func=None,
                      restraints_kwargs=None,
                      solvent:str='HOH',
                      fix_system:bool=False,
                      restart:bool=False,
                      dcd_output:bool=False,
                      pdp_output:bool=False,
                      data_output:bool=False
                      ):
    '''This is a top-level function to performe a openMM simulation based on 
    a .gro structur file and a .top topology file. This process includes a 
    equilibration of the system.
    
    Parameters 
    ----------
        forcefield: str, directory name of the force field folder, which needs
                    to be included in the working directory
        
        equisteps: int, number of integration steps for equilibrium
        
        nsteps: int, number of integration steps
        
        integrator_scheme: str, acronym of integration scheme
                           implemented schemes class: LangevinSplittingGirsanov
        
        integrator_splitting: str, if splitting scheme is chosen the order is to
                              be given with 'R' = A, 'V' = B and 'O' = O.
        
        nstxout: int, defines the write out frequency of the simulations output
        
        outputMD: str, name of the output folder, default: 'unbiasedMD'
        
        inputMD: str, name of the input folder, default: 'inputMD'
        
        plumedMD: str, name of the plumed folder, default: 'plumedMD'
        
        PlatformByNamen: str, Get the registered Platform with a particular 
                         name. See OpenMM documentation/Platform
        
        gro_input: str, name of starting structure in .gro format for simulation 
                   default: 'box_ions'
        
        top_input: str, name of initial topology of system, in .top format
                   default: 'topol'
        
        plumed_file: str, name of plumed input file, default: 'plumedCommittor.dat'
                   
        colvar_file: str, name of file to store collectiv variable via PLUMED, 
                     default: 'COLVAR'
        
        equilibration_integrator_scheme: str, acronym of integration scheme
                                         acronym for implemented schemes:
                                         'Langevin', 'LangevinWithGirsanov', 'GSD'
                                         default : 'LangevinWithGirsanov' 
        
        equilibration_integrator_splitting: str (optional), if splitting scheme 
                                           is chosen the order is to be given 
                                           with 'R' = A, 'V' = B and 'O' = O.
                                           default : 'R V O V R'
                                           
        temperature: float, openmm.unit.Quantity compatible with kelvin, 
                     default: 300.0*unit.kelvin Fictitious "bath" temperature

        collision_rate:  float, openmm.unit.Quantity compatible with 1/picoseconds, 
                         default: 2.0/unit.picoseconds Collision rate
                         
        timestep:  float, openmm.unit.Quantity compatible with femtoseconds, 
                   default: 1.0*unit.femtoseconds Integration timestep
                   
        nonbondedMethod: func, The method to use for nonbonded interactions. 
                         Allowed values are NoCutoff, CutoffNonPeriodic, 
                         CutoffPeriodic, Ewald, PME, or LJPME. 
                         see documentation openMM/forcefield/createSystem
                         default: openmm.app.PME
                         
        nonbondedCutoff: func, The cutoff distance to use for nonbonded interactions,
                         openmm.unit.Quantity compatible with nanometer,
                         default: 0.6*units.nanometer
                         
        constraints: func, Specifies which bonds and angles should be implemented 
                     with constraints. Allowed values are None, HBonds, 
                     AllBonds, or HAngles. 
                     see documentation openMM/forcefield/createSystem
                     default: openmm.app.HBonds
                     
        restraints_func: func, CustomExternalForce restraining the position of particles
                         only option is restraints_harmonic_force, analog to 
                         https://openmm.github.io/openmm-cookbook/dev/notebooks/restraints_constraints_forces/
        
        restraints_kwargs: dict, specifies input for restraints_func
        
        solvent: str, atoms of solvent, default: 'HOH'
        
        fix_system: bool, extra functionality, setting the masses of the simulation
                    system to zero, default: False
                    
        restart: bool, if True restarts a unbiased simulation, where chckpt file exists
                    
        dcd_output: bool, if True the dcd trajectory is printed out, default: False
        
        pdp_output: bool, if True the pdp trajectory is printed out, default: False
        
        data_output: bool, if True an output file for data (temperature, kinetic, 
                     potential, steps) of the trajectory are printed out, 
                     default: False
            

        References
        ----------
            OpenMM Python API : http://docs.openmm.org/latest/api-python/
            [Schäfer, Keller 2024] Implementation of Girsanov reweighting

        -------
            # Import
            >>> from reweightingtools.simulation import openMM_simulation, restraints_harmonic_force
            >>> from openmm.unit import nanometer
        
            # define MD input 
            >>> forcefield='amber99.ff'
            >>> equisteps= 5000
            >>> nsteps= 500000000
            >>> integrator_scheme= 'Langevin'
            >>> integrator_splitting='R V O V R'
            >>> nstxout=100
            >>> solvent='HOH'
        
            # define restrains
            >>> restraints_kwargs=dict()   
            >>> restraints_kwargs['multiple_restraints']=True
            >>> restraints_kwargs['atom_name']=[('CA'),('I')] #('CA','CL')
            >>> restraints_kwargs['force_constants']=[200000.0,500000.0]
            >>> restraints_kwargs['restraint_x']=[True, True]
            >>> restraints_kwargs['restraint_y']=[True, True]
            >>> restraints_kwargs['restraint_z']=[False, True]

            # Run openMM simulation
            >>> openMM_simulation(forcefield,
                                  equisteps,
                                  nsteps,
                                  integrator_scheme,
                                  integrator_splitting,
                                  nstxout,
                                  restraints_func= [restraints_harmonic_force,restraints_harmonic_force],
                                  restraints_kwargs=restraints_kwargs,
                                  fix_system=True
                                  )
        
    '''
    # setup directory structure for MD
    input_directory, output_directory, forcefield_directory, bias_directory = setup_MD_directories(forcefield,
                                                                                                   inputMD,
                                                                                                   outputMD, 
                                                                                                   plumedMD)

    # load structure, topology file and define platform
    gro, top, platform = get_grotopplatform(PlatformByNamen,
                                            forcefield_directory,
                                            input_directory,
                                            output_directory,
                                            gro_input,
                                            top_input
                                            )
    if restart:        
        positions = None
    else:
        # equilibration
        positions = make_equilibration(top, 
                                       gro, 
                                       platform, 
                                       temperature, 
                                       equisteps,
                                       equilibration_integrator_scheme=equilibration_integrator_scheme,
                                       equilibration_integrator_splitting=equilibration_integrator_splitting,
                                       collision_rate=collision_rate,
                                       timestep=timestep,
                                       nonbondedMethod=nonbondedMethod,
                                       nonbondedCutoff=nonbondedCutoff,
                                       constraints=constraints,
                                       solvent=solvent,
                                       fix_system=fix_system
                                       )

    # plumed input   
    if plumed_file in os.listdir(bias_directory): 
        plumedCommittor = open(bias_directory+plumed_file, 'r')
        plumedCommittor = plumedCommittor.read()
        plumedCommittor = plumedCommittor%(output_directory+colvar_file,nstxout) #nstxout,
    else:
        sys.exit("The plumedCommittor.dat file for the COLVAR report doesn't exist.")

    # simulation
    make_simulation(positions, 
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
                    restart=restart,
                    dcd_output=dcd_output,
                    pdp_output=pdp_output,
                    data_output=data_output
                    )

def openMM_biased_simulation(forcefield:str,
                             equisteps:int,
                             nsteps:int,
                             integrator_scheme:str,
                             integrator_splitting:str,
                             nstxout:int,
                             outputMD:str='biasedMD/',
                             inputMD:str='inputMD/',
                             plumedMD:str='plumedMD/',
                             PlatformByNamen:str='CPU',
                             gro_input:str='box_ions', # ToDo: change gro and inpcrd input
                             top_input:str='topol',
                             inpcrd_input:str='ala_wat',
                             prmtop_input:str='ala_wat',
                             hills_file:str='HILLS',
                             plumed_file:str='plumedBias.dat', #ToDo: change how to include committor, now hard coded
                             externalForce_file:str='bias.txt',
                             colvar_file:str='COLVAR',
                             equilibration_integrator_scheme:str='Langevin',
                             equilibration_integrator_splitting:str='R V O V R',
                             temperature:float=300*kelvin, 
                             collision_rate:float=2*(picoseconds)**(-1),
                             timestep:float=1*femtoseconds,
                             nonbondedMethod=PME,
                             nonbondedCutoff=0.6*nanometer,
                             constraints=HBonds,
                             restraints_func=None,
                             restraints_kwargs=None,
                             solvent:str='HOH',
                             Committor=False,
                             inpcrdprmtop=False,
                             fix_system:bool=False,
                             restart:bool=False,
                             ETA:bool=False,
                             dcd_output=False,
                             pdp_output=False,
                             data_output=False
                             ):
    '''This is a top-level function to performe a biased openMM simulation based 
    on a .gro structur file and a .top topology file. Biasing the potential enegry 
    surface is performed using the PLUMED interface openmmplumed or with a string   
    defining an external force. The process includes a equilibration of the system. 
    #ToDo: In addition, the function checks whether the plumed input has been 
    written and can create a bias.plumed.#
    
    Parameters 
    ----------
        forcefield: str, directory name of the force field folder, which needs
                    to be included in the working directory
        
        equisteps: int, number of integration steps for equilibrium
        
        nsteps: int, number of integration steps
        
        integrator_scheme: str, acronym of integration scheme
                           implemented schemes class: LangevinSplittingGirsanov
        
        integrator_splitting: str, if splitting scheme is chosen the order is to
                              be given with 'R' = A, 'V' = B and 'O' = O.
        
        nstxout: int, defines the write out frequency of the simulations output
        
        outputMD: str, name of the output folder, default: 'biasedMD'
        
        inputMD: str, name of the input folder, default: 'inputMD'
        
        plumedMD: str, name of the plumed folder, default: 'plumed'
        
        PlatformByNamen: str, Get the registered Platform with a particular 
                         name. See OpenMM documentation/Platform
        
        gro_input: str, name of starting structure in .gro format for simulation 
                   default: 'box_ions'
        
        top_input: str, name of initial topology of system, in .top format
                   default: 'topol'
                   
        inpcrd_input: str, name of starting structure in .inpcrd format for simulation 
                      default:'ala_wat',
        
        prmtop_input: str, name of initial topology of system, in .prmtop format
                      default: 'ala_wat',
                   
        hills_file: str, name of hills file wich contains the bias generated with
                    with PLUMED, default: 'HILLS'
                
        plumed_file: str, name of plumed input file, default: 'plumedBias.dat'
        
        externalForce_file: str, name of text file defing the bias force as string, 
                            default: 'bias.txt'
        
        colvar_file: str, name of file to store collectiv variable via PLUMED, 
                     default: 'COLVAR'
        
        equilibration_integrator_scheme: str, acronym of integration scheme
                                         acronym for implemented schemes:
                                         'Langevin', 'LangevinWithGirsanov', 'GSD'
                                         default : 'LangevinWithGirsanov' 
        
        equilibration_integrator_splitting: str (optional), if splitting scheme 
                                           is chosen the order is to be given 
                                           with 'R' = A, 'V' = B and 'O' = O.
                                           default : 'R V O V R'
                                           
        temperature: float, openmm.unit.Quantity compatible with kelvin, 
                     default: 300.0*unit.kelvin Fictitious "bath" temperature

        collision_rate:  float, openmm.unit.Quantity compatible with 1/picoseconds, 
                         default: 2.0/unit.picoseconds Collision rate
                         
        timestep:  float, openmm.unit.Quantity compatible with femtoseconds, 
                   default: 1.0*unit.femtoseconds Integration timestep
                   
        nonbondedMethod: func, The method to use for nonbonded interactions. 
                         Allowed values are NoCutoff, CutoffNonPeriodic, 
                         CutoffPeriodic, Ewald, PME, or LJPME. 
                         see documentation openMM/forcefield/createSystem
                         default: openmm.app.PME
                         
        nonbondedCutoff: func, The cutoff distance to use for nonbonded interactions,
                         openmm.unit.Quantity compatible with nanometer,
                         default: 0.6*units.nanometer
                         
        constraints: func, Specifies which bonds and angles should be implemented 
                     with constraints. Allowed values are None, HBonds, 
                     AllBonds, or HAngles. 
                     see documentation openMM/forcefield/createSystem
                     default: openmm.app.HBonds
                     
        restraints_func: func, CustomExternalForce restraining the position of particles
                         only option is restraints_harmonic_force, analog to 
                         https://openmm.github.io/openmm-cookbook/dev/notebooks/restraints_constraints_forces/
        
        restraints_kwargs: dict, specifies input for restraints_func
        
        solvent: str, atoms of solvent, default: 'HOH'
        
        fix_system: bool, extra functionality, setting the masses of the simulation
                    system to zero, default: False
                    
        restart: bool, if True restarts a unbiased simulation, where chckpt file exists
                    
        dcd_output: bool, if True the dcd trajectory is printed out, default: False
        
        pdp_output: bool, if True the pdp trajectory is printed out, default: False
        
        data_output: bool, if True an output file for data (temperature, kinetic, 
                     potential, steps) of the trajectory are printed out, 
                     default: False
            
        References
        ----------
            OpenMM Python API : http://docs.openmm.org/latest/api-python/
            [Schäfer, Keller 2024] Implementation of Girsanov reweighting

        Example
        -------
            # Import
            >>> from reweightingtools.simulation import openMM_biased_simulation, restraints_harmonic_force
            >>> from openmm.unit import nanometer
            
            # MD input
            >>> forcefield='amber99.ff'
            >>> equisteps= 5000
            >>> nsteps= 5000000000
            >>> integrator_scheme= 'LangevinWithGirsanov'
            >>> integrator_splitting='R V O V R'
            >>> nstxout=100
            >>> solvent='HOH'

            >>> restraints_kwargs=dict()   
            >>> restraints_kwargs['multiple_restraints']=True
            >>> restraints_kwargs['atom_name']=[('CA'),('CL')] 
            >>> restraints_kwargs['force_constants']=[200000.0,500000.0]
            >>> restraints_kwargs['restraint_x']=[True, True]
            >>> restraints_kwargs['restraint_y']=[True, True]
            >>> restraints_kwargs['restraint_z']=[False, True]

            # Run openMM simulation
            >>> openMM_biased_simulation(forcefield,
                                         equisteps,
                                         nsteps,
                                         integrator_scheme,
                                         integrator_splitting,
                                         nstxout,
                                         gro_input='ClCaCl',
                                         plumed_file='plumedCOLVAR.dat',
                                         externalForce_file='bias_05z.txt',
                                         restraints_func= [restraints_harmonic_force,restraints_harmonic_force], 
                                         restraints_kwargs=restraints_kwargs,
                                         fix_system=True,
                                         #restart=True,
                                         dcd_output=False,
                                         pdp_output=False,
                                         data_output=False
                                         )
        
    '''
    
    # setup directory structure for MD
    input_directory, output_directory, forcefield_directory, bias_directory = setup_MD_directories(forcefield,
                                                                                                   inputMD,
                                                                                                   outputMD, 
                                                                                                   plumedMD)
    # ToDo: change gro and inpcrd input
    # load structure, topology file and define platform
    if inpcrdprmtop:
        gro, top, platform = get_inpcrdprmtopplatform(PlatformByNamen,
                                                      forcefield_directory,
                                                      input_directory,
                                                      output_directory,
                                                      inpcrd_input,
                                                      prmtop_input
                                                     )
    else:
        gro, top, platform = get_grotopplatform(PlatformByNamen,
                                                forcefield_directory,
                                                input_directory,
                                                output_directory,
                                                gro_input,
                                                top_input
                                                )
            
    if restart:        
        positions = None
    else:
        # equilibration
        positions = make_equilibration(top, 
                                       gro, 
                                       platform, 
                                       temperature, 
                                       equisteps,
                                       equilibration_integrator_scheme=equilibration_integrator_scheme,
                                       equilibration_integrator_splitting=equilibration_integrator_splitting,
                                       collision_rate=collision_rate,
                                       timestep=timestep,
                                       nonbondedMethod=nonbondedMethod,
                                       nonbondedCutoff=nonbondedCutoff,
                                       constraints=constraints,
                                       solvent=solvent,
                                       fix_system=fix_system
                                       )

    # bias input   
    if externalForce_file in os.listdir(bias_directory):
        plumedScript = open(bias_directory+plumed_file, 'r')
        plumedScript = plumedScript.read()
        if Committor:
            plumedScript = plumedScript%(nstxout,output_directory+colvar_file,nstxout) 
        else:
            plumedScript = plumedScript%(output_directory+colvar_file,nstxout)
        externalForce=bias_directory+externalForce_file
        print(externalForce)
        if ETA:
            make_externalForce_simulation_ETA(positions, 
                                              np.array([0.5,0.5,0.5]), 
                                              top, 
                                              gro, 
                                              plumedScript,
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
                                              restart=restart,
                                              dcd_output=dcd_output,
                                              pdp_output=pdp_output,
                                              data_output=data_output
                                              )
        else:
            make_externalForce_simulation(positions, 
                                      top, 
                                      gro, 
                                      plumedScript,
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
                                      restart=restart,
                                      dcd_output=dcd_output,
                                      pdp_output=pdp_output,
                                      data_output=data_output
                                     )
        sys.exit("The simulation with external force is finished.")
    else:
        if hills_file in os.listdir(bias_directory): 
            if plumed_file in os.listdir(bias_directory): 
                # read the bias via PLUMED
                plumedForce = open(bias_directory+plumed_file, 'r')
                plumedForce = plumedForce.read()
                plumedForce = plumedForce%(nstxout, bias_directory+hills_file,output_directory+colvar_file,nstxout) 
                
                # simulation
                make_biased_simulation(positions, 
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
                           restart=restart,
                           dcd_output=dcd_output,
                           pdp_output=pdp_output,
                           data_output=data_output
                          )
                sys.exit("The simulation with plumed bias is finished.")
        else:
            #ToDo: run short METAD to create HILLS file see openMM_metadynamics
            sys.exit("The HILLS file for the bias doesn't exist.")
