#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 15:27:38 2023

@author: schaefej51
This is a collection of wrapper functions for deeptime tasks.
"""
from deeptime.markov.msm import MaximumLikelihoodMSM
from deeptime.markov import TransitionCountEstimator
from deeptime.markov import GirsanovReweightingEstimator
from deeptime.markov.msm._markov_state_model import MarkovStateModel
from deeptime.markov.tools.estimation import largest_connected_set
import numpy as np
    
def get_implied_timescales(dtraj,
                            lagtimes,
                            k,
                            countmode,
                            reversible,
                            stationary_distribution_constraint,
                            reweighting_factors
                            ):
    '''This function is a low-level function for evaluating the implied time scales 
    based on a discrete trajectory and a count estimator. A choice can be made 
    between the TransitionCountEstimator or the GirsanovReweightingEstimator.'''
    # instantiate the MaximumLikelihoodMSM estimator for MSM 
    estimator = MaximumLikelihoodMSM(reversible=reversible,
                                     stationary_distribution_constraint=stationary_distribution_constraint)
    # create MSM models to analyse for timescales
    models = []
    for lagtime in lagtimes:
        if reweighting_factors != None:
            # collect statistics from discret trajectory using GirsanovReweightingEstimator
            reweighted_count_estimator = GirsanovReweightingEstimator(lagtime=lagtime,
                                                                      count_mode=countmode)
            
            # use Girsanov reweighting estimator to fit a reweighted count model
            reweighted_counts = reweighted_count_estimator.fit(data=dtraj,
                                                               reweighting_factors=reweighting_factors).fetch_model()  
            # re-estimate a reweighted MSM based on reweighted count model
            reweighted_msm = estimator.fit(reweighted_counts).fetch_model()
            # create MSM model based on reweighted transition matrix
            reweighted_msm_model = MarkovStateModel(reweighted_msm.transition_matrix, 
                                                    lagtime=lagtime,
                                                    n_eigenvalues=k)
            # collect the timescales frome the reweighted MSM model
            models.append(reweighted_msm_model.timescales())
        else:
            # collect statistics from discret trajectory using deeptime’s transition count estimator
            count_estimator = TransitionCountEstimator(lagtime=lagtime, 
                                                       count_mode=countmode)
            # use transition count estimator to fit a count model
            counts = count_estimator.fit(dtraj).fetch_model() 
            # re-estimate a MSM based on the count model
            msm = estimator.fit(counts).fetch_model()
            # create MSM model based on transition matrix
            msm_model = MarkovStateModel(msm.transition_matrix, 
                                         lagtime=lagtime,
                                         n_eigenvalues=k)
            # collect timescales from the MSM model
            models.append(msm_model.timescales())
    its_data = np.array(models)
    return its_data   

def get_eigenvectors(dtraj,
                     lagtime,
                     reversible,
                     stationary_distribution_constraint,
                     countmode,
                     k,
                     reweighting_factors,
                     right=False
                     ):
    '''This function is a low-level function for evaluating the implied time scales 
    based on a discrete trajectory and a count estimator. A choice can be made 
    between the TransitionCountEstimator or the GirsanovReweightingEstimator.'''
    # instantiate the MaximumLikelihoodMSM estimator for MSM 
    estimator = MaximumLikelihoodMSM(
            reversible=reversible,
            stationary_distribution_constraint=stationary_distribution_constraint)

    if reweighting_factors!= None:
        # collect statistics from discret trajectory using GirsanovReweightingEstimator
        reweighted_count_estimator = GirsanovReweightingEstimator(lagtime=lagtime,
                                                                  count_mode=countmode)
        # use Girsanov reweighting estimator to fit a reweighted count model
        counts = reweighted_count_estimator.fit(data=dtraj,
                                                reweighting_factors=reweighting_factors).fetch_model()  

    else:
        # collect statistics from discret trajectory using deeptime’s transition count estimator
        count_estimator = TransitionCountEstimator(lagtime=lagtime, 
                                                   count_mode=countmode)
        # use transition count estimator to fit a count model
        counts = count_estimator.fit(dtraj).fetch_model() 



    # re-estimate a MSM based on the count model
    msm = estimator.fit(counts).fetch_model()
    # create MSM model based on transition matrix
    msm_model = MarkovStateModel(msm.transition_matrix, 
                                 lagtime=lagtime,
                                 n_eigenvalues=k)
    
    if right:
        # collect right eigenvectors from the MSM model
        eigenvecs = msm_model.eigenvectors_right(k)
    else:
        # collect left eigenvectors from the MSM model
        eigenvecs = msm_model.eigenvectors_left(k)
    # store the largest connected sets
    lcs = largest_connected_set(counts.count_matrix)
    return eigenvecs, lcs
   
def get_eigenvalues(dtraj,
                     lagtime,
                     reversible,
                     stationary_distribution_constraint,
                     countmode,
                     k,
                     reweighting_factors
                     ):
    '''This function is a low-level function for evaluating the implied time scales 
    based on a discrete trajectory and a count estimator. A choice can be made 
    between the TransitionCountEstimator or the GirsanovReweightingEstimator.'''
    # instantiate the MaximumLikelihoodMSM estimator for MSM 
    estimator = MaximumLikelihoodMSM(
            reversible=reversible,
            stationary_distribution_constraint=stationary_distribution_constraint)

    if reweighting_factors!= None:
        # collect statistics from discret trajectory using GirsanovReweightingEstimator
        reweighted_count_estimator = GirsanovReweightingEstimator(lagtime=lagtime,
                                                                  count_mode=countmode)
        # use Girsanov reweighting estimator to fit a reweighted count model
        counts = reweighted_count_estimator.fit(data=dtraj,
                                                reweighting_factors=reweighting_factors).fetch_model()  

    else:
        # collect statistics from discret trajectory using deeptime’s transition count estimator
        count_estimator = TransitionCountEstimator(lagtime=lagtime, 
                                                   count_mode=countmode)
        # use transition count estimator to fit a count model
        counts = count_estimator.fit(dtraj).fetch_model() 



    # re-estimate a MSM based on the count model
    msm = estimator.fit(counts).fetch_model()
    # create MSM model based on transition matrix
    msm_model = MarkovStateModel(msm.transition_matrix, 
                                 lagtime=lagtime,
                                 n_eigenvalues=k)
    
    # collect eigenvalues from the MSM model
    eigenval = msm_model.eigenvalues(k)
    # store the largest connected sets
    lcs = largest_connected_set(counts.count_matrix)
    return eigenval, lcs
