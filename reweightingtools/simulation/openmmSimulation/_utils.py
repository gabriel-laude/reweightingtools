#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:38:58 2023

@author: schaefej51
"""
import numpy as np
from datetime import datetime as dati
import os
import sys

# openmm_utils
def setup_MD_directories(forcefield,inputMD,outputMD,plumedMD):
    ''' This is a low level function to check if MD directory set up is correct.
    Arguments:
        forcefield: string; path to forcefield files
        inputMD: string; name for input folder for MD simulation (e.g. .top .gro)
        outputMD: string; name for output folder of MD simulation 
                  (e.g. .dcd ReweightingFactors.txt)
        plumedMD: string; name for PLUMED input folder for MD simulation 
                  (e.g. plumedBIAS.dat)
    '''
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
        #ToDo: create plumed input
        #os.mkdir(bias_directory)
        sys.exit("Directory for PLUMED input doesn't exist")
     
    forcefield=forcefield +'/'  
    forcefield_directory=os.path.join(cwd,forcefield)
    if os.path.isdir(forcefield_directory):
        if not os.listdir(forcefield_directory):
            print("Force field directory is empty.")
    else:
        sys.exit("Force field directory doesn't exist")
    
    return input_directory, output_directory, forcefield_directory, bias_directory

def logging_MDparameters(output_directory,
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
                         plumedForces,
                         externalForce,
                         restart=False
                         ):
    ''' This is a low level function to print the MD parameters in the MD.log file.
    Arguments:
        output_directory: string; path to MD output folder
        integrator_scheme: string; name of integration scheme
        nsteps: int; number of integration steps
        nstxout: int; write out frequency
        temperature: float; simulation temperature
        collision_rate: float; colision rate for stochastic integrator 
        timestep: float; integration timestep
        integrator_splitting: string; sequence of update functions
        nonbondedMethod: func, The method to use for nonbonded interactions
        nonbondedCutoff: func, The cutoff distance to use for nonbonded interactions
        constraints: func, Specifies which bonds and angles should be implemented 
                     with constraints
        plumedForces: string; file name for plumed input                
    '''
    log = open(output_directory + "/MD.log", 'a')
    if restart:
        log.write('##               _____________________________________________________________________\n' )
        log.write('## R E S T A R T E D \n' )                                    
        log.write('##               _____________________________________________________________________\n' )
    else:
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
    if plumedForces != None:
        log.write('##               _____________________________________________________________________\n' )
        log.write(plumedForces)
        log.write('##               _____________________________________________________________________\n' )
    if externalForce != None:
        log.write('##               _____________________________________________________________________\n' )
        log.write(externalForce)
        log.write('##               _____________________________________________________________________\n' )
    log.write("Simulation Start " + str(dati.now()) + "\n" )
    log.close();    





