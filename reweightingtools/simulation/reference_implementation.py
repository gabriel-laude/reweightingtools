#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Jan 31 13:14:04 2023

@author: schaefej51
"""
import numpy as np
import types

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
            
        gamma        = xi * m
        sqrt         = np.sqrt( 2 * kB * T/gamma * time_step)  

        q            = np.zeros([n_steps + 1, position_inital.size])  
        q[0,:]       = position_inital
        f            = []
        
        for k in range(n_steps):
            q[k + 1,:] = q[k,:] - time_step/gamma * potential_gradient(q[k,:]) + sqrt * eta[k,:] 
            f.append(-potential_gradient(q[k + 1]))
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
        
        mkBT           = m * kB * T
        t2m            = time_step / (2 * m)
        t2             = time_step / 2
        exp            = np.exp(-xi * time_step)
        sqrt           = np.sqrt(mkBT * (1 - exp**2))
        
        q              = np.zeros([n_steps + 1, position_inital.size]) 
        p              = np.zeros([n_steps + 1, momentum_inital.size])  

        q[0,:], p[0,:] = position_inital, momentum_inital
        f          = []
        
        for k in range(n_steps):
            q[k + 1,:] = q[k,:] + t2m * p[k,:] 
            p[k + 1,:] = p[k,:] - t2 * potential_gradient(q[k + 1,:])
            p[k + 1,:] = exp * p[k + 1,:] + sqrt * eta[k,:]
            p[k + 1,:] = p[k + 1,:] - t2 * potential_gradient(q[k + 1,:])
            f.append(-potential_gradient(q[k + 1]))
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
            
class Simulation(object):
    def __init__(self, 
                 potential_gradient, 
                 position_inital,
                 momentum_inital,
                 n_steps, 
                 time_step, 
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
        
    #def __call__(self, **kwargs):
    #    ''' '''
    #    return self.func(**kwargs) 
    
    
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
    
    def simulate_batches(self, 
                         _integration_scheme, 
                         batch_size, 
                         write_out_frequency,
                         #file_name='traj', 
                         store_directory='./'
                         ):
        log = open(store_directory + "positions.txt", 'a')
        log.write('## %s trajectory in position space\n')
        np.savetxt(log,  np.array([[self.position_inital[0],self.position_inital[1]]])) #, fmt='%1.3f', newline=", ")
        log.close()
        
        if _integration_scheme.__name__ != 'Euler_Maruyama':
            log = open(store_directory + "velocities.txt", 'a')
            log.write('## %s trajectory in momentum space\n')
            np.savetxt(log,  np.array([[self.momentum_inital[0],self.momentum_inital[1]]])) #, fmt='%1.3f', newline=", ")
            log.close()
        
        if type(self.eta) == np.ndarray:
            if self.eta.shape[0]!=self.n_steps:
                print('the ramdom nuber sequence loaded does not match the number of integration steps')

        n_epochs     = int(self.n_steps / batch_size)
        self.n_steps = batch_size
        if type(self.eta) != np.ndarray:
            saveEta = True
        else:
            ETA=self.eta.reshape(n_epochs,batch_size,self.position_inital.shape[0])
            saveEta = False
        
        for i in range(n_epochs):
            if saveEta is False:
                self.eta = ETA[i]
            if _integration_scheme.__name__ == 'Euler_Maruyama':
                q_traj, eta, f = Simulation.integration_scheme(self, _integration_scheme)
                
                self.position_inital = q_traj[-1]
                log = open(store_directory + "positions.txt", 'a')
                np.savetxt(log, q_traj[1::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
                
                log = open(store_directory + "force.txt", 'a')
                np.savetxt(log, f[1::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
                
                #if saveEta:
                log = open(store_directory + "eta.txt", 'a')
                log.write('## %s trajectory of random number\n')
                np.savetxt(log, eta[::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
            
            
            else:
                q_traj, p_traj, eta, f = Simulation.integration_scheme(self, _integration_scheme)
                
                self.position_inital = q_traj[-1]
                log = open(store_directory + "positions.txt", 'a')
                np.savetxt(log, q_traj[1::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
            
                self.momentum_inital = p_traj[-1]
                log = open(store_directory + "velocities.txt", 'a')
                np.savetxt(log, p_traj[1::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
                
                log = open(store_directory + "force.txt", 'a')
                np.savetxt(log, f[1::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
                
                log = open(store_directory + "eta.txt", 'a')
                log.write('## %s trajectory of random number\n')
                np.savetxt(log, eta[::write_out_frequency]) #, fmt='%1.3f', newline=", ")
                log.write("\n")
                log.close()
