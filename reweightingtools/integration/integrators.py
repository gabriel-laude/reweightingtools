#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Collection of Langevin integration schemes.

@author: J.-L. Schaefer
"""
import numpy as np
#import types

# ToDo: make it comparable with the openmm output, analog to ABOBA

def Euler_Maruyama(potential_gradient,
                   position_inital,  
                   n_steps, 
                   time_step, 
                   m,  # ATTENTION IF YOU CHOSE DIFF MASS V = p/m
                   xi, 
                   kB, 
                   T,
                   eta=None):
        ''' 
        Euler-Maruyama is implemented according to Leimkuhler, Molecular Dynamics 
        Chapter 6.3.4 Overdamped Limit of Langevin Dynamics Eq. 6.36
        '''
        if type(position_inital) is not np.ndarray:
            position_inital = np.ndarray(position_inital)
        if eta is None:
            print('The random numbers are generated.')
            eta = np.random.normal(0,1,size=(n_steps, position_inital.size))
            # to make it comparable with the openmm output 
            eta = np.vstack([np.zeros((1,position_inital.size)), eta])
            
        gamma        = xi * m
        sqrt         = np.sqrt( 2 * kB * T/gamma * time_step)  

        q            = np.zeros([n_steps + 1, position_inital.size])  
        q[0,:]       = position_inital
        f[0,:]         = -potential_gradient(q[0,:]) # to make it comparable with the openmm output set to zero
        
        for k in range(n_steps):
            q[k + 1,:] = q[k,:] - time_step/gamma * potential_gradient(q[k,:]) + sqrt * eta[k,:] 
            f[k + 1,:] = -potential_gradient(q[k + 1])
        return q , eta , np.array(f)


        
def GSD(potential_gradient, 
        position_inital,
        momentum_inital,
        n_steps, 
        time_step,  
        m,
        xi, 
        kB, 
        T,
        eta=None):
        '''  
        GROMACS Stochastic Dynamics integrator (GSD) is implemented according 
        to https://arxiv.org/pdf/2204.02105.pdf
        '''
        if type(position_inital) is not np.ndarray:
            position_inital = np.ndarray(position_inital)
        if type(momentum_inital) is not np.ndarray:
            momentum_inital = np.ndarray(momentum_inital)
        if eta is None:
            print('The random numbers are generated.')
            eta = np.random.normal(0,1,size=(n_steps, position_inital.size)) 
            # to make it comparable with the openmm output 
            eta = np.vstack([np.zeros((1,position_inital.size)), eta])
        
        #m, xi, kB, T   = para.values()
        mkBT           = m * kB * T
        f              = 1.0 - np.exp(-xi * time_step)
        sqrt           = np.sqrt(f * (2 - f) * mkBT)
        
        q              = np.zeros([n_steps + 1, position_inital.size]) 
        p              = np.zeros([n_steps + 1, momentum_inital.size])  
        q[0,:], p[0,:] = position_inital, momentum_inital
        
        for k in range(n_steps):
            if type(potential_gradient) == types.MethodType: 
                p[k + 1,:] = p[k,:] - potential_gradient(q[k,:]) * time_step
                Deltap     = -f * p[k + 1,:] + sqrt * eta[k] 
                q[k + 1,:] = q[k] + (p[k + 1,:]/m + Deltap/(2 * m)) * time_step
                p[k + 1,:] = p[k + 1,:] + Deltap 
            else:
                p[k + 1,:] = p[k,:] - potential_gradient[k,:] * time_step
                Deltap     = -f * p[k + 1] + sqrt * eta[k] 
                q[k + 1,:] = q[k] + (p[k + 1,:]/m + Deltap/(2 * m)) * time_step
                p[k + 1,:] = p[k + 1,:] + Deltap 
                
        return q, p, eta

def BAOA(potential_gradient, 
         position_inital,
         momentum_inital,
         n_steps, 
         time_step,   
         m,
         xi, 
         kB, 
         T,
         eta=None):
        '''  
        BAOA integrator is implemented according to
        https://arxiv.org/pdf/2204.02105.pdf, compare supplementary information
        https://arxiv.org/src/2204.02105v1/anc/Supplementary_Information.pdf
        '''
        if type(position_inital) is not np.ndarray:
            position_inital = np.ndarray(position_inital)
        if type(momentum_inital) is not np.ndarray:
            momentum_inital = np.ndarray(momentum_inital)
        if eta is None:
            print('The random numbers are generated.')
            eta = np.random.normal(0,1,size=(n_steps, position_inital.size)) 
            # to make it comparable with the openmm output 
            eta = np.vstack([np.zeros((1,position_inital.size)), eta])
        
        #m, xi, kB, T   = para.values()    
        mkBT           = m * kB * T
        t2m            = time_step / (2 * m)
        exp            = np.exp(-xi * time_step)
        sqrt           = np.sqrt(mkBT * (1 - exp**2))
        
        q              = np.zeros([n_steps + 1, position_inital.size]) 
        p              = np.zeros([n_steps + 1, momentum_inital.size])  
        q[0,:], p[0,:] = position_inital, momentum_inital
        
        for k in range(n_steps):
            if type(potential_gradient) == types.MethodType: 
                p[k + 1,:] = p[k,:] - time_step * potential_gradient(q[k,:])
                q[k + 1,:] = q[k,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k,:]
                q[k + 1,:] = q[k + 1,:] + t2m * p[k + 1,:] 
            else:
                p[k + 1,:] = p[k,:] - time_step * potential_gradient[k,:]
                q[k + 1,:] = q[k,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k,:]
                q[k + 1,:] = q[k + 1,:] + t2m * p[k + 1,:] 
                
        return q, p, eta   

def ABOBA(potential_gradient, 
          position_inital,
          momentum_inital,
          n_steps, 
          time_step, 
          m,  # ATTENTION IF YOU CHOSE DIFF MASS V = p/m
          xi, 
          kB, 
          T,
          eta=None):
        '''  
        ABOBA integrator is implemented according to
        https://arxiv.org/pdf/2204.02105.pdf, compare supplementary information
        https://arxiv.org/src/2204.02105v1/anc/Supplementary_Information.pdf
        '''
        if type(position_inital) is not np.ndarray:
            position_inital = np.ndarray(position_inital)
        if type(momentum_inital) is not np.ndarray:
            momentum_inital = np.ndarray(momentum_inital)
        if eta is None:
            print('The random numbers are generated.')
            eta = np.random.normal(0,1,size=(n_steps, position_inital.size)) 
            # to make it comparable with the openmm output 
            eta = np.vstack([np.zeros((1,position_inital.size)), eta])
        
        mkBT           = m * kB * T
        t2m            = time_step / (2 * m)
        t2             = time_step / 2
        exp            = np.exp(-xi * time_step)
        sqrt           = np.sqrt(mkBT * (1 - exp**2))
        
        q              = np.zeros([n_steps + 1, position_inital.size]) 
        p              = np.zeros([n_steps + 1, momentum_inital.size]) 
        f	       = np.zeros([n_steps + 1, momentum_inital.size]) 

        q[0,:], p[0,:] = position_inital, momentum_inital
        f[0,:]         = -potential_gradient(q[0,:])
        
        for k in range(n_steps):
            q[k + 1,:] = q[k,:] + t2m * p[k,:] 
            p[k + 1,:] = p[k,:] - t2 * potential_gradient(q[k + 1,:])
            p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k+1,:]
            p[k + 1,:] = p[k + 1,:] - t2 * potential_gradient(q[k + 1,:])
            f[k + 1,:] = -potential_gradient(q[k + 1]) #f.append(-potential_gradient(q[k + 1]))
            q[k + 1,:] = q[k + 1,:] + t2m * p[k + 1,:] 

        return q, p, eta, np.array(f)
    
def BAOAB(potential_gradient, 
          position_inital,
          momentum_inital,
          n_steps, 
          time_step,  
          m,
          xi, 
          kB, 
          T,  
          eta=None):
        '''  
        BAOAB integrator is implemented according to
        https://arxiv.org/pdf/2204.02105.pdf, compare supplementary information
        https://arxiv.org/src/2204.02105v1/anc/Supplementary_Information.pdf
        '''
        if type(position_inital) is not np.ndarray:
            position_inital = np.ndarray(position_inital)
        if type(momentum_inital) is not np.ndarray:
            momentum_inital = np.ndarray(momentum_inital)
        if eta is None:
            print('The random numbers are generated.')
            eta = np.random.normal(0,1,size=(n_steps, position_inital.size)) 
            # to make it comparable with the openmm output 
            eta = np.vstack([np.zeros((1,position_inital.size)), eta])
        
        #m, xi, kB, T   = para.values() 
        mkBT           = m * kB * T
        t2m            = time_step / (2 * m)
        t2             = time_step / 2
        exp            = np.exp(-xi * time_step)
        sqrt           = np.sqrt(mkBT * (1 - exp**2))
        
        q              = np.zeros([n_steps + 1, position_inital.size]) 
        p              = np.zeros([n_steps + 1, momentum_inital.size])  
        q[0,:], p[0,:] = position_inital, momentum_inital
        
        for k in range(n_steps):
            if type(potential_gradient) == types.MethodType: 
                p[k + 1,:] = p[k,:] - t2 * potential_gradient(q[k,:])
                q[k + 1,:] = q[k,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k,:]
                q[k + 1,:] = q[k + 1,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = p[k + 1,:] - t2 *potential_gradient(q[k + 1,:])
            else:
                p[k + 1,:] = p[k,:] - t2 * potential_gradient[k,:]
                q[k + 1,:] = q[k,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k,:]
                q[k + 1,:] = q[k + 1,:] + t2m * p[k + 1,:] 
                p[k + 1,:] = p[k + 1,:] - t2 *potential_gradient[k + 1,:]
            
            return q, p, eta
