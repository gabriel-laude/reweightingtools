#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 31 13:42:48 2023

@author: schaefej51

This file contains function for eigenvalue and vector analysis of a MSM with deeptime.
"""
from reweightingtools.analysis.deeptime_wrapper import *
from reweightingtools.analysis._utils import *
from openmm.unit import * 
from openmmtools.constants import kB

def deeptime_implied_timescales(lagtimes:list,
                                gridsize:int=100,
                                number_eigenvectors:int=4,
                                countmode:str="sliding",
                                reversible=False,
                                stationary_distribution_constraint=None,
                                reweighting=False,
                                its_file:str='/its',
                                analysisMD:str='analysis_MD',
                                dtraj_file:str='/discretized',
                                gF_file:str='/g_factors',
                                MF_file:str='/M_factors',
                                temperature:float=300*kelvin, 
                                save_its=False
                                ):
    '''This is a top-level function to evaluate the implied timescales for a 
    number of lag times based on a discretized trajectory, and in case on 
    pre-reweighting factors using deeptime.
    
    Parameters 
    ----------
        lagtimes: list (array-like with integers), integer lag times at which 
                  the implied timescales will be calculated. 
        
        gridsize: int, corresponding to discretisation, here number of bins
        
        number_eigenvectors: int, number of implied timescales to be computed.
        
        countmode: str, mode to obtain count matrices from discrete trajectories. 
                   default: ’sliding’ : A trajectory of length T will have T−tau
                   counts at time indexes
        
        reversible: bool, estimate transition matrix reversibly (True) 
        or nonreversibly (False)
        
        stationary_distribution_constraint: None or array-like, sample from 
        the given stationary distribution
        
        reweighting: bool, if True pre-reweighting factors are loaded and 
                     reweighted MSM is constructed 
        
        its_file: str, name of the its output file, default: '/its'
        
        analysisMD: str, name of the output folder, default: 'analysis_MD'
        
        dtraj_file: str, name of the file containing list of discretized 
                    trajectories, default: '/discretized'
        
        gF_file: str, name of the file containing the list of reweighting factor 
                 g trajectories, default: '/g_factors'
        
        MF_file: str, name of the file containing the list of reweighting factor 
                 M trajectories, default: '/M_factors'
                                           
        temperature: float, openmm.unit.Quantity compatible with kelvin, 
                     default: 300.0*unit.kelvin Fictitious "bath" temperature
                     
        save_its: bool, if True the its array will be saved in analysisMD folder
        
        References
        ----------
            OpenMM Python API : http://docs.openmm.org/latest/api-python/
            [Schaefer, Keller 2024] Implementation of Girsanov reweighting in OpenMM and Deeptime

        Example
        -------
            # Import
            >>> from reweightingtools.analysis.discretisation import *
            >>> from reweightingtools.analysis.api import *
        
            # Set input
            >>> cwd='/home/project/'
            >>> analysisMD=cwd+'analysis_MD' 
            >>> gridsize=100
            >>> reversible=True
            >>> nstxout=100
            >>> lagtimes=[1,3,5,7,9,10] 
            >>> max2D, min2D = np.array([ 1.5 ,  3.5]), np.array([-3.5, -1.5])
            >>> traj  = np.load(MD_directory+'/traj.npy') 
            >>> dtraj = trajectory2D_MBP(traj,
                                         gridsize,
                                         min2D,
                                         max2D)
            >>> np.save(analysisMD+'/discretized_'+str(gridsize),dtraj)
        
            # Prepare directory, eg. /discretized, containing discretized 
            # trajectory, g and M factor output from openMM simulation using 
            # the LangevinSplittingGirsanov integrator
´            >>> deeptime_implied_timescales(lagtimes=lagtimes,
                                             gridsize=gridsize,
                                             reversible=reversible,
                                             reweighting=True,
                                             its_file='/its_rwght',
                                             analysisMD=analysisMD,
                                             dtraj_file='/discretized',
                                             gF_file='/g_factors',
                                             MF_file='/M_factors',
                                             save_its=True)
        
            # Visualize reweighted implied timescales 
            >>> its_rwght = np.load(analysisMD+'/its_rwght_'+str(gridsize)+'.npy')
            >>> i= 0 # slowest process
            >>> plt.plot(np.array(lagtimes),its_rwght[:,i],label='rwght')
        
    '''
    k=number_eigenvectors
    # MD_directory=setup_MD_directory(outputMD)
    analysis_directory=setup_ANA_directory(analysisMD)
    # load discretized trajectory 
    discretized=np.load(analysis_directory+dtraj_file+'_'+str(gridsize)+'.npy')
    discretized=[discretized]
    # load pre-reweighting factors
    if reweighting:
        gF=np.load(analysis_directory+gF_file+'.npy',allow_pickle=True)
        gF=[gF]
        MF=np.load(analysis_directory+MF_file+'.npy',allow_pickle=True)
        MF=[MF]
        if len(discretized) == len(gF) and len(gF)==len(MF):
            pass
        else:
            sys.exit('Input dimensionality is not correct for dtraj, g or M factor.')
        # get reweighting factors
        beta = 1 / kB / temperature / 0.001
        for i, U in enumerate(gF):
            gF[i] = np.exp(beta*U)  
        for i,logM in enumerate(MF):
            MF[i] = logM 
        reweighting_factors=(gF,MF)
        # get reweighted implied timescales
        its=get_implied_timescales(dtraj=discretized,
                                           lagtimes=lagtimes,
                                           k=k,
                                           countmode=countmode,
                                           reversible=reversible,
                                           stationary_distribution_constraint=stationary_distribution_constraint,
                                           reweighting_factors=reweighting_factors)
    else:
        print('No reweighting factors will be used in the analysis.')
        # get implied timescales
        its=get_implied_timescales(dtraj=discretized,
                                    lagtimes=lagtimes,
                                    k=k,
                                    countmode=countmode,
                                    reversible=reversible,
                                    stationary_distribution_constraint=stationary_distribution_constraint,
                                    reweighting_factors=None)
    if save_its:
        np.save(analysis_directory+its_file+'_'+str(gridsize),its)
    else:
        return its

def deeptime_eigenvectors(lagtime:int,
                                gridsize:int=10,
                                number_eigenvectors:int=4,
                                countmode:str="sliding",
                                reversible=False,
                                stationary_distribution_constraint=None,
                                reweighting=False,
                                analysisMD:str='analysis_MD',
                                eigenvector_file:str='/evecs',
                                dtraj_file:str='/discretized',
                                gF_file:str='/g_factors',
                                MF_file:str='/M_factors',
                                temperature:float=300*kelvin
                               ):
    '''This is a top-level function to evaluate the eigenvectors for a lag time and 
    a discretized trajectory, and in case on pre-reweighting factors using deeptime.
    
    Parameters 
    ----------
        lagtimes: list (array-like with integers), integer lag times at which 
                  the implied timescales will be calculated. 
        
        gridsize: int, corresponding to discretisation, here number of bins
        
        number_eigenvectors: int, number of implied timescales to be computed.
        
        countmode: str, mode to obtain count matrices from discrete trajectories. 
                   default: ’sliding’ : A trajectory of length T will have T−tau
                   counts at time indexes
        
        reversible: bool, estimate transition matrix reversibly (True) 
        or nonreversibly (False)
        
        stationary_distribution_constraint: None or array-like, sample from 
        the given stationary distribution
        
        reweighting: bool, if True pre-reweighting factors are loaded and 
                     reweighted MSM is constructed 
        
        its_file: str, name of the its output file, default: '/its'
        
        analysisMD: str, name of the output folder, default: 'analysis_MD'
        
        dtraj_file: str, name of the file containing list of discretized 
                    trajectories, default: '/discretized'
        
        gF_file: str, name of the file containing the list of reweighting factor 
                 g trajectories, default: '/g_factors'
        
        MF_file: str, name of the file containing the list of reweighting factor 
                 M trajectories, default: '/M_factors'
                                           
        temperature: float, openmm.unit.Quantity compatible with kelvin, 
                     default: 300.0*unit.kelvin Fictitious "bath" temperature
        
        References
        ----------
            OpenMM Python API : http://docs.openmm.org/latest/api-python/
            [Schäfer, Keller 2024] Implementation of Girsanov reweighting
        Example
        -------
            # Import
            >>> from reweightingtools.analysis.api import *

            # Prepare directory, with 
            # - discretized trajectory
            # - g and M factor from openMM simulation using LangevinSplittingGirsanov integrator
´           
            >>> deeptime_eigenvectors(lagtime=150 ,
                                      gridsize=36,
                                      number_eigenvectors=2,
                                      reversible=True,
                                      reweighting=True
                                      )
    '''
    k=number_eigenvectors
    #MD_directory=setup_MD_directory(outputMD)
    analysis_directory=setup_ANA_directory(analysisMD)
    #%%
    # load discretized trajectory 
    discretized=np.load(analysis_directory+dtraj_file+'.npy')
    discretized=[discretized]
    # load pre-reweighting factors
    if reweighting:
        gF=np.load(analysis_directory+gF_file+'.npy',allow_pickle=True)
        gF=[gF]
        MF=np.load(analysis_directory+MF_file+'.npy',allow_pickle=True)
        MF=[MF]
        if len(discretized) == len(gF) and len(gF)==len(MF):
            pass
        else:
            print(len(discretized), len(gF), len(MF))
            sys.exit('Input dimensionality is not correct for dtraj, g or M factor.')
        # get reweighting factors
        beta = 1 / kB / temperature / 0.001
        for i, U in enumerate(gF):
            gF[i] = np.exp(beta*U)  
        for i,logM in enumerate(MF):
            MF[i] = logM 
        reweighting_factors=(gF,MF)
        # get reweighted eigenvectors
        evecs,lcs=get_eigenvectors(discretized,
                                   lagtime,
                                     reversible,
                                     stationary_distribution_constraint,
                                     countmode,
                                     k,
                                     reweighting_factors)
        
        np.save(analysis_directory+eigenvector_file, evecs)
        np.save(analysis_directory+eigenvector_file+'_lcs', lcs)
    else:
        print('No reweighting factors will be used in the analysis.')
        # get eigenvectors
        evecs,lcs=get_eigenvectors(discretized,
                               lagtime,
                               reversible,
                               stationary_distribution_constraint,
                               countmode,
                               k,
                               reweighting_factors=None)

        
        np.save(analysis_directory+eigenvector_file, evecs)
        np.save(analysis_directory+eigenvector_file+'_lcs', lcs)
