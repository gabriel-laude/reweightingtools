#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of 2D bias functions to perform python simulatios or to calculate reweighting factors.   
'''

import numpy as np
from abc import ABC, abstractmethod
from .potentials1D import Linear as Linear1D
from .potentials1D import SmoothEnvelope, Gaussian

#def D2_linearXbiasX_gradient(x, k):
#        y = np.zeros(x.shape)
#        return np.array([k * np.ones(x.shape),y])
        
#def D2_linearXbiasX(x, k):
#        y = np.zeros(x.shape)
#        return np.array([k * x, y])
        
class D2bias(ABC):
    @abstractmethod    
    def __init__(self, param): 
        pass
    @abstractmethod    
    def potential(self,x_line, y_line):
        pass
    @abstractmethod
    def gradient(self,x_line, y_line):
        pass
    ## ToDo inseart plotting stuff
    #@abstractmethod
    #def plot(self, x_line):
    #    return plotPotentialsGradients(x_line, self.potential, self.gradient) 


class Linear(D2bias):
    '''linear potential: V(x,y)=kx*(x-x0) + ky*(y-y0); with strenght k=[kx,ky], displacement [x0,y0].'''
    def __init__(self, strength=[1.0,0.0], displacement=[0.0,0.0]):
        self.linear1D_x = Linear1D(strength=strength[0],displacement=displacement[0])
        self.linear1D_y = Linear1D(strength=strength[1],displacement=displacement[1])
     
    def potential(self,x_line, y_line):
        pot =  self.linear1D_x.potential(x_line) + self.linear1D_y.potential(y_line)
        return pot

    def gradient(self,x_line, y_line):
        grad = np.array([self.linear1D_x.gradient(x_line), self.linear1D_y.gradient(y_line)])
        return grad

class GausEnvelope(D2bias):
    '''linear potential: V(x,y)=kx*(x-x0) + ky*(y-y0); with strenght k=[kx,ky], displacement [x0,y0], xbounds=None, ybounds =[min,max] to set on pot, xsmooth or ysmooth=[strengh, slope, downsize].'''
    def __init__(self, xbounds, ybounds, strength=[1.0,1.0]):
        self.smooth_x = SmoothEnvelope(xbounds[0], xbounds[1])
        self.smooth_y = SmoothEnvelope(ybounds[0], ybounds[1])
        self.gaussian_x = Gaussian(strength=strength[0], mu=(xbounds[0]+xbounds[1])/2, sigma=1.)
        self.gaussian_y = Gaussian(strength=strength[1], mu=(ybounds[0]+ybounds[1])/2, sigma=1.)

     
    def potential(self,x_line, y_line):
        pot=self.gaussian_x.potential(x_line)*self.smooth_y.potential(y_line) + self.gaussian_y.potential(y_line) * self.smooth_x.potential(x_line)
        return pot

    def gradient(self,x_line, y_line):
        #grad = np.array([self.linear1D_x.gradient(x_line), self.linear1D_y.gradient(y_line)])
        #return grad
        pass
               
class Harmonic(D2bias):
    '''linear potential: V(x)=k *(x-x0); with strenght k, displacement x0 (or list for [x,y]), dim specify the dimension along bias is applied'''
    def __init__(self, springConst=[1.0,0], displacement=[0.0,0.0]):
        
        self.potential1D_x = Harmonic(springConst=springConst[0],displacement=displacement[0])
        self.potential1D_y = Harmonic(springConst=springConst[1],displacement=displacement[1])

    def potential(self,x_line, y_line):
        pot_x = self.potential1D_x.potential(x_line)
        pot_y = self.potential1D_y.potential(y_line)
        return np.sum(np.array([pot_x, pot_y]), axis=0)
    def gradient(self,x_line, y_line):
        grad_x = self.potential1D_x.gradient(x_line)
        grad_y = self.potential1D_y.gradient(y_line)
        return np.array([grad_x, grad_y])
             
class Polynomial(D2bias):
    '''Polynomial potential: V(x)=k * sum^m_i c_i x^(i); with strenght k, dim specify the dimension along bias is applied'''
    def __init__(self, strength=[1.0,0]):
    # ToDo: generalize parameters and sum function
        self.a = 2.05777925e-02
        self.b = -2.31737460e-02
        self.c = 3.83236939e-03  
        self.d = 3.91666334e-02
        self.e = -1.39425035e-02 
        self.f = -3.38910809e-02
        self.g = 3.81706681e-04
        self.h = 1.24385771e-02
        self.i = 3.36985889e-03
        self.j = -1.17944563e-03
        self.k = -7.50078923e-04
        self.l = -1.37745928e-04
        self.m =  -8.81084849e-06
        
        self.k_x, self.k_y  = strength[0], strength[1]
        
    def sum_p(self, x): 
        return (self.a + self.b * x + self.c * x ** 2 + self.d * x ** 3 + self.e * x ** 4 + self.f * x ** 5 + self.g * x ** 6 + self.h * x ** 7+ self.i * x ** 8 + self.j * x ** 9+ self.k * x ** 10 + self.l * x ** 11+ self.m * x ** 12)
    def sum_g(self, x):
        return (self.b + 2 * self.c * x + 3 * self.d * x ** 2 + 4* self.e * x ** 3 + 5* self.f * x ** 4 + 6* self.g * x ** 5 + 7* self.h * x ** 6+ 8* self.i * x ** 7 + 9* self.j * x ** 8+10* self.k * x ** 9 + 11* self.l * x ** 10+ 12* self.m * x ** 11)
            
    def potential(self,x_line, y_line):
        return np.array([-self.k_x * self.sum_p(x_line), -self.k_y * self.sum_p(y_line)])
    def gradient(self,x_line, y_line):
        return np.array([-self.k_x * self.sum_g(x_line) , -self.k_y * self.sum_g(y_line)])
