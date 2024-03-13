#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb 14 10:17:54 2023

@author: schaefej51
"""
import numpy as np

class MBLP():
    """OpenMM custom force for propagation on the coupled Mueller Brown potential.
    Includes evaluation of the potential energy surface and its gradienr, as
    well as bias examples.
    """
   
    def __init__(self): #, lamb): #, **para): # here parameters hard coded
        #self.para = para 
        self.A_n = np.array([[-20,-10],[-17,1.5]])
        self.a_n = np.array([[-1,-1],[-6.5,0.7]])
        self.b_n = np.array([[0,0],[11,0.6]])
        self.c_n = np.array([[-10,-10],[-6.5,0.7]])
        self.x_n = np.array([[1,0],[-0.5,-1]])
        self.y_n = np.array([[0.,0.5],[1.5,1]])
        self.N = np.arange(self.A_n.shape[0])
        
        super(MBLP, self).__init__()
        
    def summand(self, l, n, x, y):
        return self.A_n[l,n] * np.exp(self.a_n[l,n] * (x - self.x_n[l,n])**2 +\
                                 self.b_n[l,n] * (x - self.x_n[l,n]) * (y - self.y_n[l,n]) +\
                                         self.c_n[l,n] * (y - self.y_n[l,n])**2)

    def mueller_brown(self, l, x, y):
        return sum(MBLP.summand(self, l, n, x, y) for n in self.N)
       
    def coupled_mueller_brown(self, x, y):
        return MBLP.mueller_brown(self, 0, x, y) +\
                 MBLP.mueller_brown(self, 1, x, y)
    
    def v_prime_X(self, l, n, x, y):
        return 2 * self.a_n[l,n] * (x - self.x_n[l,n]) +\
                self.b_n[l,n] * (y - self.y_n[l,n])
    
    def mueller_brown_gradient_X(self, l, x, y):
        return sum(MBLP.v_prime_X(self, l, n, x, y) *\
                   MBLP.summand(self, l, n, x, y) for n in self.N)
    
    def v_prime_Y(self, l, n, x, y):
        return self.b_n[l,n] * (x - self.x_n[l,n]) +\
                2 * self.c_n[l,n] * (y - self.y_n[l,n])
            
    def mueller_brown_gradient_Y(self, l, x, y):
            return sum(MBLP.v_prime_Y(self,l, n, x, y) *\
                       MBLP.summand(self, l, n, x, y) for n in self.N)
    
    def coupled_mueller_brown_gradient(self, vector_position):
        x, y = vector_position
        return np.array([MBLP.mueller_brown_gradient_X(self, 0, x, y) +\
             MBLP.mueller_brown_gradient_X(self, 1, x, y),\
             MBLP.mueller_brown_gradient_Y(self, 0, x, y) +\
             MBLP.mueller_brown_gradient_Y(self, 1, x, y)])

    def biasX_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([2.5 ,0.0])
    
    def bias1X_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([1 ,0.0])
    
    def bias05X_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([0.5,0.0])
    
    def bias25Y_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([0.0,2.5])
    
    def bias1Y_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([0.0,1])
    
    def bias05Y_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([0.0,0.5])
    
    def bias025X_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([0.25 ,0.0])
    
    def bias5X_gradient(self, vector_position):
        x, y = vector_position
        return MBLP.coupled_mueller_brown_gradient(self, vector_position) +\
            np.array([5 ,0.0])

def biasX_gradient_simple(x):
        y = np.zeros(x.shape)
        return np.array([2.5*np.ones(x.shape),y])
        
def biasX_simple(x):
        y = np.zeros(x.shape)
        return np.array([2.5* x,y])
