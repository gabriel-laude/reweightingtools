#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Jul 27 14:54:14 2023

@author: schaefej51
"""
from ._utils import *

def trajectory1D(traj1D,
                 gridsize
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
                 trajxyzmax,
                 return_dxdy=False
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

    if return_dxdy:
        return dtraj.astype(int), dx, dy
    else:
        return dtraj.astype(int)

def cluster_centers_1D(traj1D, gridsize):
    min_traj = np.min(traj1D)
    traj1D_shifted = traj1D - min_traj
    dx = np.max(traj1D_shifted) / (gridsize - 1)
    centers = np.arange(gridsize) * dx + min_traj + dx / 2
    
    return centers

def calculate_cluster_centers(trajxyz, dtraj, gridsize, trajxyzmin):
    # Find unique microstates
    unique_microstates = np.unique(dtraj)

    # Initialize a dictionary to accumulate points for each microstate
    microstate_points = {microstate: [] for microstate in unique_microstates}

    # Populate the dictionary with points corresponding to each microstate
    for i, microstate in enumerate(dtraj):
        microstate_points[microstate].append(trajxyz[i])

    cluster_centers = []
    for microstate, points in microstate_points.items():
        points = np.array(points)
        center_x = np.mean(points[:, 0])
        center_y = np.mean(points[:, 1])
        cluster_centers.append((center_x, center_y))
    
    assert len(cluster_centers) == len(unique_microstates)

    return np.array(cluster_centers), unique_microstates
