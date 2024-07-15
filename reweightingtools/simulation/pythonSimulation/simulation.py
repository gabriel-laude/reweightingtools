#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:14:04 2023

@author: schaefej51

This file contains a class to perform 2D python MD simulations.
"""
import os 
import sys
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = os.getcwd()
dir_path = os.path.join(dir_path, os.pardir) 
module_dir = os.path.abspath(dir_path)
sys.path.insert(0, module_dir)

# Now you can import your module as usual
from reweightingtools.integration.integrators import *


import numpy as np
import h5py
import time

class Simulation(object):
    '''The class MD simulation via python 2D integrators. Output trajectories are saved ether as txt 
    or h5py file.
    Set up simulation according to  reweightingtools.simulation.pythonSimulation.templates
    '''
    def __init__(self, 
                 potential_gradient, 
                 position_inital,
                 momentum_inital,
                 n_steps, 
                 time_step, 
                 potential_class,
                 mass, 
                 friction, 
                 Boltzmann, 
                 temperature,
                 eta=None):

        self.potential_gradient = potential_gradient
        self.position_inital    = position_inital
        
        if momentum_inital is None:
            print('Select the initial pulse, otherwise the value zero is set for all directions.')
            self.momentum_inital =  np.zeros(position_inital.shape) 
        else:
            self.momentum_inital    = momentum_inital
        
        self.potential_class = potential_class
        self.n_steps   = n_steps
        self.time_step = time_step
        self.eta       = eta
        self.m = mass 
        self.xi = friction 
        self.kB = Boltzmann 
        self.T = temperature
        
    def __repr__(self):
        return ''' Returns one dimensional (q=x) numerical integrators: 
        EulerMaruyama: x, 
        GSD,BAOA,ABOBA,BAOAB: x,p
    Returns:
        q (numpy.ndarray(n_steps + 1)): configuraiton trajectory
        p (numpy.ndarray(n_steps + 1)): momentum trajectory
    '''
    def __str__(self):
        return '''ARGS:
        potential_gradient (function): computes the forces of a single configuration
        position_inital (numpy.ndarray(n, d)): initial configuration
        momentum_init (numpy.ndarray(n, d)): initial momenta
        n_steps (int): number of integration steps
        time_step (float): time step for the integration
        mass (numpy.ndarray(n)): particle masses, DEFAULT==1.0
        friction_const (float): damping term, use zero if not coupled, DEFAULT==1.0
        beta (float): inverse temperature, DEFAULT==1.0
        Boltzmann_const (float): Boltzmann constant k_B, DEFAULT==1.0
        temperature (float): temperature T, DEFAULT==1.0
        share_eta (boolean): define eta as global (True) or local (False), DEFAULT==FALSE'''
    
    def integration_scheme(self, _integration_scheme):
        valids = ('Euler_Maruyama', 'GSD', 'ABOBA', 'BAOA', 'BAOAB')
        if _integration_scheme.__name__ not in valids:
            raise ValueError("Invalid integration scheme \"{}\", possible values are {}.".format(str(_integration_scheme), valids))
        if _integration_scheme.__name__ == 'Euler_Maruyama':
            return _integration_scheme(self.potential_gradient, 
                                          self.position_inital,
                                          self.n_steps, 
                                          self.time_step,  
                                          self.m,
                                          self.xi, 
                                          self.kB, 
                                          self.T,  
                                          self.eta,
                                          )
        else: 
            return _integration_scheme(self.potential_gradient, 
                                          self.position_inital,
                                          self.momentum_inital,
                                          self.n_steps, 
                                          self.time_step,  
                                          self.m,
                                          self.xi, 
                                          self.kB, 
                                          self.T,  
                                          self.eta,
                                          )
                                        
    def logging(self, _integration_scheme, write_out_frequency, batch_size):
        loggings={
            'integration scheme' : _integration_scheme.__name__,
            'number of steps' : self.n_steps,
            'time step' : self.time_step,
            'write out frequency' : write_out_frequency,
            'batch size' : batch_size,
            'potential' : self.potential_class.__class__,
            'potential class' : self.potential_class.__dict__,
            'position inital' : self.position_inital,
            'momentum inital' : self.momentum_inital,
            'mass' : self.m,
            'friction constant' : self.xi,
            'Boltzmann' : self.kB,
            'temperature' : self.T,
            'random number' : self.eta
        }
        return loggings
    def write_log(quantity, store_directory, file_name,  mode='a'):
        if not isinstance(quantity, str):
            quantity = str(quantity)
        log = open(store_directory + file_name+".txt", mode)
        log.write(quantity)
        log.write("\n")
        log.close()

    def write_txt(quantity, store_directory, file_name,  mode='a'):
        log = open(store_directory + file_name+".txt", mode)
        np.savetxt(log, quantity) #, fmt='%1.3f', newline=", ")
        #log.write("\n")
        log.close()

    def write_h5py(quantity, quantity_name, hdf):
        hdf.create_dataset(quantity_name, data=quantity)
    
    def write_quantity(quantity, quantity_name, h5py_format, hdf=None, store_directory=None):
        if h5py_format:
            Simulation.write_h5py(quantity, quantity_name=quantity_name, hdf=hdf)
        else:
            Simulation.write_txt(quantity, file_name=quantity_name, store_directory=store_directory) 

    def batch(self, _integration_scheme, write_out_frequency, nbatch, h5py_format, position_name="position", force_name="force", random_name="random", momentum_name="momentum", **kwargs):
        if h5py_format:
            position_name="position"+nbatch
            force_name="force"+nbatch
            random_name="random"+nbatch
            momentum_name="momentum"+nbatch
        if _integration_scheme.__name__ == 'Euler_Maruyama':
            q, eta, f = Simulation.integration_scheme(self, _integration_scheme)
            self.position_inital = q[-1]
            Simulation.write_quantity(quantity = q[1::write_out_frequency], quantity_name=position_name, h5py_format=h5py_format, **kwargs)
            Simulation.write_quantity(quantity = f[1::write_out_frequency], quantity_name=force_name, h5py_format=h5py_format, **kwargs)
            Simulation.write_quantity(quantity = eta[::write_out_frequency], quantity_name=random_name, h5py_format=h5py_format, **kwargs)  
        else:
            q, p, eta, f = Simulation.integration_scheme(self, _integration_scheme)
            self.position_inital = q[-1]
            self.momentum_inital = p[-1]
            Simulation.write_quantity(quantity = q[1::write_out_frequency], quantity_name=position_name, h5py_format=h5py_format, **kwargs)
            Simulation.write_quantity(quantity = p[1::write_out_frequency], quantity_name=momentum_name, h5py_format=h5py_format, **kwargs)
            Simulation.write_quantity(quantity = f[1::write_out_frequency], quantity_name=force_name, h5py_format=h5py_format, **kwargs)
            Simulation.write_quantity(quantity = eta[::write_out_frequency], quantity_name=random_name, h5py_format=h5py_format, **kwargs)  
    
    def epochs(self, _integration_scheme, batch_size, write_out_frequency, **kwargs):
        n_epochs     = int(self.n_steps / batch_size)
        self.n_steps = batch_size
        if type(self.eta) != np.ndarray:
            saveEta = True
        else:
            ETA=self.eta.reshape(n_epochs, batch_size, self.position_inital.shape[0])
            saveEta = False
        for i in range(n_epochs):
            if saveEta is False:
                self.eta = ETA[i]
            Simulation.batch(self, _integration_scheme=_integration_scheme, write_out_frequency=write_out_frequency, nbatch=str(i), **kwargs)
            
    def simulate(self, _integration_scheme, write_out_frequency, batch_size, **kwargs):
        Simulation.write_quantity(quantity = np.array([[self.position_inital[0],self.position_inital[1]]]), quantity_name="position", **kwargs)
        if _integration_scheme.__name__ != 'Euler_Maruyama':
            Simulation.write_quantity(quantity = np.array([[self.momentum_inital[0],self.momentum_inital[1]]]), quantity_name="momentum", **kwargs)
        if type(self.eta) == np.ndarray:
            if self.eta.shape[0]!=self.n_steps:
                print('the ramdom nuber sequence loaded does not match the number of integration steps')
        if batch_size is not None:
            Simulation.epochs(self, _integration_scheme=_integration_scheme, batch_size=batch_size, write_out_frequency=write_out_frequency, **kwargs)
        else:
            Simulation.batch(self, _integration_scheme=_integration_scheme, write_out_frequency=write_out_frequency, nbatch=str(1), **kwargs)

    def run(self, h5py_format, _integration_scheme, write_out_frequency, store_directory, batch_size, position_name='position',):
        loggings = Simulation.logging(self, _integration_scheme, write_out_frequency, batch_size)
        for key, value in loggings.items():
            if isinstance(value, dict):
                for k, v in value.items():
                    write = str(k)+' '+str(v)+"\n"
                    Simulation.write_log(quantity=write, store_directory=store_directory, file_name='log')
            else: 
                write = str(key)+' '+str(value)+"\n"
                Simulation.write_log(quantity=write, store_directory=store_directory, file_name='log')
        Simulation.write_log(quantity=time.ctime(), store_directory=store_directory, file_name='log')
        if h5py_format:
            with h5py.File(store_directory+'h5py_trajs'+'.h5', 'w') as hdf: 
                kwargs = {'hdf' : hdf,
                          'h5py_format' : h5py_format}
                Simulation.simulate(self, _integration_scheme=_integration_scheme, write_out_frequency=write_out_frequency, batch_size=batch_size, **kwargs)
        else:
            kwargs = {'store_directory' : store_directory, 
            'h5py_format': h5py_format}
            Simulation.simulate(self, _integration_scheme=_integration_scheme, write_out_frequency=write_out_frequency, batch_size=batch_size, **kwargs)
