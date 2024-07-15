'''This file contains reader functions for different python MD output formats.
'''
import numpy as np
import h5py
import time

def read_h5py_traj(path, file_name='h5py_trajs'):
    '''To read a h5py file, extract trajectories for position, momentum, force and random number.
    '''
    pos=[]
    mom=[]
    foc=[]
    ran=[]
    with h5py.File(path+file_name+'.h5', 'r') as hdf:
        ls = list(hdf.keys())
        for traj in ls:
            if 'position' in traj:
                pos.append(np.array(hdf.get(traj)))
            if 'momentum' in traj:
                mom.append(np.array(hdf.get(traj)))
            if 'force' in traj:
                foc.append(np.array(hdf.get(traj)))
            if 'random' in traj:
                ran.append(np.array(hdf.get(traj)))
    position = np.array(pos[1:])
    position = np.concatenate(position, axis=0)
    momentum = np.array(mom[1:])
    momentum = np.concatenate(momentum, axis=0)
    force = np.array(foc)
    force = np.concatenate(force, axis=0)
    random = np.array(ran)
    random = np.concatenate(random, axis=0)
    return position, momentum, force, random


def read_h5py_trajs(n_trajs, path, file_name='h5py_trajs'):
    '''Concatenate h5py trajectories from different simulation runs.
    Arguments:
        n_trajs: int of different simulation runs
        path: string the path to output directory
    '''
    if isinstance(n_trajs, int):
        traj_range = range(n_trajs)
    elif isinstance(n_trajs, list):
        traj_range = n_trajs
    else:
        raise ValueError('n_trajs should give a list of indecies or output directories or a int that gives the maximum for the folder indices but it gives: ' + str(n_trajs))
    pos=[]
    mom=[]
    foc=[]
    ran=[]
    for i in traj_range:
        position, momentum, force, random = read_h5py_trajs(path=path+str(i)+'/', file_name=file_name)
        pos.append(position)
        mom.append(momentum)
        foc.append(force)
        ran.append(random)
    position = np.array(pos)
    position = np.concatenate(position, axis=0)
    momentum = np.array(mom)
    momentum = np.concatenate(momentum, axis=0)
    force = np.array(foc)
    force = np.concatenate(force, axis=0)
    random = np.array(ran)
    random = np.concatenate(random, axis=0)
    
def read_txt_trajs(n_trajs, path, file_name):
    '''Concatenate txt trajectories from different simulation runs.
    Arguments:
        n_trajs: int of different simulation runs
        path: string the path to output directory
    '''
    if isinstance(n_trajs, int):
        traj_range = range(n_trajs)
    elif isinstance(n_trajs, list):
        traj_range = n_trajs
    else:
        raise ValueError('n_trajs should give a list of indecies or output directories or a int that gives the maximum for the folder indices but it gives: ' + str(n_trajs))
    pos=[]
    for i in range(n_trajs): 
        pos.append(np.loadtxt(path+str(i)+file_name+'.txt'))
    position = np.array(pos)
    position = np.concatenate(position, axis=0)