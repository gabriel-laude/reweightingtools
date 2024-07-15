#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:54:14 2023

@author: schaefej51
"""
from ._utils import *

def trajectory1D(traj1D,
                 gridsize,
                 min1D,
                 max1D
                 ):
    ''' Low level function. This function discretizes a number of trajectories
    into microstates. The grid is defined according to a given (minv, max) 
    boundary.'''
    traj1D = traj1D - np.min(traj1D)
    dx = np.max(traj1D)- np.min(traj1D)
    dx = dx/(gridsize-1)
    dtraj = np.floor(traj1D/dx)
    return dtraj.astype(int)

def trajectory2D(trajxyz,
                 gridsize,
                 trajxyzmin,
                 trajxyzmax
                 ):
    ''' Top level function. This function discretizes a number of trajectories
    into microstates. The grid is defined on the single trajectory or according
    to a given (minv, max) boundary.'''
    trajxyz[:,0] = trajxyz[:,0] - np.min(trajxyz[:,0])
    trajxyz[:,1] = trajxyz[:,1] - np.min(trajxyz[:,1])
    dx = np.max(trajxyz[:,0])- np.min(trajxyz[:,0])
    dy = np.max(trajxyz[:,1])- np.min(trajxyz[:,1])
    dx = dx/(gridsize-1) 
    dy = dy/(gridsize-1) 
    trajxyz[:,0] = np.floor(trajxyz[:,0]/dx)
    trajxyz[:,1] = np.floor(trajxyz[:,1]/dy)
    dtraj = trajxyz[:,0]*gridsize  + trajxyz[:,1]
    return dtraj.astype(int)