#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Jul 03 09:32:38 2024

@author: schaefej51

This file contains functions to compute reweighting factors for python simulations.
"""

import numpy as np

def M_ABOBA(fU, eta, timestep, T, m, xi, kB):
    '''Computes path reweighting factor for ABOBA scheme based on [Kieninger Keller 2023]
    '''
    h = timestep/1
    sigma = np.sqrt(T * kB / m) 
    b = np.sqrt(1 - np.exp(- 2 * xi * h))
    a = np.exp(- xi * h)
    DeltaEta0 = 1 / (b * sigma * m) * (1 + a) * timestep / 2 
    DeltaEta0 = DeltaEta0 * fU
    logM = eta * DeltaEta0 + 0.5 * (DeltaEta0 * DeltaEta0)
    return DeltaEta0, logM
