#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 17 14:38:58 2023

@author: schaefej51

This file includes functions to read, write or get infos from openMM MD simulations.
"""
import mdtraj as md #ToDo. move to utils
import numpy as np
from datetime import datetime as dati
import os
import sys
import matplotlib.pyplot as plt

def setup_MD_directory(outputMD):
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
    if os.path.isdir(outputMD):
        MD_directory=outputMD
    elif os.path.isdir(os.path.join(cwd,outputMD)):
        MD_directory=os.path.join(cwd,outputMD)
    else:
        sys.exit("Directory for MD output not found")
    return MD_directory

def setup_ANA_directory(analysisMD):
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
    if os.path.isdir(analysisMD):
        analysis_directory=analysisMD
    elif os.path.isdir(os.path.join(cwd,analysisMD)):
        analysis_directory=os.path.join(cwd,analysisMD) 
    else:
        os.mkdir(analysisMD)
        analysis_directory=analysisMD
    return analysis_directory

# ana_utils   
def read_times_COLVAR(output_directory, colvar_file):
    ''' This is a low level function to read in am numpy array of integration 
    steps and collective variable output.'''
    if os.path.exists(output_directory+colvar_file):
        read_data    = np.loadtxt(output_directory+colvar_file)
        colvar=[]
        for cv in range(1,read_data.shape[1]):
            colvar.append([read_data[:,0], read_data[:,cv]])
        return np.array(colvar)
    else:
        sys.exit("Directory for COLVAR output doesn't exist")

def read_RWGHTF(output_directory):
    ''' This is a low level function to read in the reweighting factors (also for 
    resarted runs).'''
    if os.path.exists(output_directory):
        read_data = np.loadtxt(output_directory+'/ReweightingFactors.txt')
        return read_data
    else:
        sys.exit("Directory for reweighting output doesn't exist")

def get_escape_data(list_escape_trajs,
                    outputMD,
                    analysisMD,
                    esc_file = '/esc_steps',
                    colvar_file ='/COLVAR',
                    save_esc=False
                    ):
    '''Function to write out the escape time steps of the trajectory. Furthermore
    one can save the minimum and maximum value of COLVAR like distance d and the 
    cordination number cn, a distance and reweighting factor container of all runs.
    The function has to be executed in the analysis directory. 
    Arguments:
        list_escape_trajs: list; list of integers that define outputMD runs
        save_esc: boolean; to save a list of all escape time steps 
    '''
    data=[]
    analysis_directory=setup_ANA_directory(analysisMD)
    for i in list_escape_trajs:
        outMD=outputMD+'_'+str(i)
        MD_directory=setup_MD_directory(outMD)
        colvar = read_times_COLVAR(MD_directory, colvar_file)
        distances = colvar[0].swapaxes(1,0)[:,1]
        # Using a PLUMED committor function
        esc_timestep = len(distances)
        output_esc=[esc_timestep]
        data.append(output_esc)
    if save_esc:
        np.save(analysis_directory+esc_file, data)
    else:
        return data

## Histogram and free energy analysis         
def _get_histogram_distances(distances, cutoff, bins, density):
    ''' To get a histogram for the sime distance values. Unpopolated states are ignored.
    Arguments:
        distances: 1D array of distances between two molecolar or atomic units.
        cutoff: maximum distance to consider.
        bins: number of bins for histogram.
        density: numpy density argument.
    '''
    hist, bin_edges = np.histogram(distances, bins=bins, density=density)
    bin_array       = bin_edges[:-1] + (bin_edges[-1] - bin_edges[-2])/2
    hist_zeros = np.where(hist==0)[0]
    try:
        missing_microstates = bin_array[hist_zeros]
        if hist_zeros.size != 0:
            for missing_microstate in missing_microstates:
                if missing_microstate < cutoff:
                    print('ATTENTION:\n Not all microstates are sampled! Here microstate %s of %s microstates is missing.' %(missing_microstate, bins))
            hist[hist_zeros]+=1  
    except:
        pass
    
    if cutoff != 0:
        hist            = hist[np.where(bin_array<cutoff)]
        bin_array       = bin_array[np.where(bin_array<cutoff)]
    
    return hist, bin_array

def get_density_Distances(Distances, cutoff, bins, density):
    ''' To get density according to list of distances (Distances).
     Arguments:
        Distances: List of distances between two molecolar or atomic units of different runs.
        cutoff: maximum distance to consider.
        bins: number of bins for histogram.
        density: numpy density argument.
    '''
    distances = np.concatenate(Distances)
    return _get_histogram_distances(distances, cutoff, bins, density)   

def get_potentialEnegry(T, density, unit):
    ''' get potential energy function according to temperatur, a density array, and units unit.
    '''
    kB  = scipy.constants.Boltzmann
    if unit=='kJ/mol':
        NA  = scipy.constants.Avogadro
        return (-T*kB*np.log(density))*(10**(-3)*NA)
    if unit=='J':
        return (-T*kB*np.log(density))
