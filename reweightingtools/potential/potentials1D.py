#!/usr/bin/env python3
# -*- coding: utf-8 -*-
''' Collection of potential functions to perform python simulatios or to calculate reweighting factors.   
'''

import numpy as np
from abc import ABC, abstractmethod
import os 
import sys
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = os.getcwd()
dir_path = os.path.join(dir_path, os.pardir) 
module_dir = os.path.abspath(dir_path)
sys.path.insert(0, module_dir)
from reweightingtools.analysis.plotting.visualizations import plot_potential_class
    
class D1potentials(ABC):

    @abstractmethod    
    def __init__(self, param): 
        pass
    @abstractmethod    
    def potential(self,x_line):
        pass
    @abstractmethod
    def gradient(self,x_line):
        pass
    @abstractmethod
    def plot(self, x_line):
        return plot_potential_class(x_line, self.__main__) 


class Linear(D1potentials):
    '''linear potential: V(x)=k *(x-x0); with strenght k, displacement x0'''
    def __init__(self, strength=1.0, displacement=0.0):
        self.k  = strength
        self.x0 = displacement
            
    def potential(self,x_line):
        return self.k * (x_line - self.x0)
    def gradient(self,x_line):
        return self.k * np.ones_like(x_line)
    
class Harmonic(D1potentials):
    '''Harmonic potential: V(x)=1/2 k (x-x0)^2; with spring constant (springConst k), displacement x0'''
    def __init__(self, springConst=1.0, displacement=0.0):
        self.k  = springConst
        self.x0 = displacement
            
    def potential(self,x_line):
        return self.k * np.power(x_line - self.x0, 2) / 2
    def gradient(self,x_line):
        return self.k * (x_line-self.x0)  
        
class Doublewell(D1potentials):
    '''double well potential: V(x)= k * ((x-a)^2-b)^2, also tilted version possible'''
    def __init__(self, strength, displace, hight, tilted=False):
        self.k=strength
        self.a=displace
        self.b=hight
        self.tilted = tilted
  
    def potential(self,x_line):
        if self.tilted:
            return self.k * np.power((np.power(x_line-self.a,2)-self.b), 2) + x_line
        else:
            return self.k * np.power((np.power(x_line-self.a,2)-self.b), 2)
    def gradient(self,x_line):
        if self.tilted:
            return 4 * self.k * (np.power(x_line-self.a,2) - self.b) * (x_line - self.a) + 1
        else:
            return 4 * self.k * (np.power(x_line-self.a,2) - self.b) * (x_line - self.a)
        
class Triplewell(D1potentials):
    '''triple well potential V(x)=k * ((x - a)^3 - b*x)^2 - x^3 + x + c, where k is the scaling faktor, a,b and c are shift parameter along x/y axis. 
    '''
    def __init__(self, hight1=4, hight2=0, hight3=3/2, shifty=2):
        self.k = hight1
        self.a = hight2
        self.b = hight3
        self.c = shifty
    
    def potential(self,x_line):
        return self.k * np.power((np.power(x_line-self.a,3)-self.b*x_line),2) - np.power(x_line,3) + x_line + self.c
    def gradient(self,x_line):
        return 2 * self.k * (3*(x_line-self.a)**2 -self.b) *((x_line-self.a)**3-self.b*x_line)-3*x_line**2+1

class Logistic(D1potentials):
    '''logistic potential: V(x)=k*1/(1+exp(-m(x-x0))); with strenght k, displacement x0 and slope m'''
    def __init__(self, strength=1.0, slope=1., displacement=0.0):
        self.k  = strength
        self.m  = slope
        self.x0 = displacement
            
    def potential(self, x_line):
        return self.k * 1 / (1 + np.exp(- (self.m * (x_line - self.x0)))) 
    def gradient(self,x_line):
        return -self.k  * (self.m * np.exp(-self.m * (x_line - self.x0))) / (np.exp(- (self.m * (x_line - self.x0))) + 1)**2 

class Gaussian(D1potentials):
    '''gaussian potential: V(x)=..; with strengh, mu,sigma'''
    def __init__(self, strength=1, mu=1.0, sigma=1.):
        self.strength=strength
        self.mu  = mu
        self.sigma  = sigma
            
    def potential(self, x_line):
        return self.strength / (self.sigma*np.sqrt(2 * np.pi)) * np.exp(-(x_line - self.mu)**2 / (2*self.sigma**2))
    def gradient(self,x_line):
        return self.strength / (self.sigma*np.sqrt(2 * np.pi)) * np.exp(-(x_line - self.mu)**2 / (2*self.sigma**2)) * (x_line - self.mu)

class SmoothEnvelope(D1potentials):
    '''sum of logistic potential V_l: V(x)=V_l(x_line, xmin) + V_l(x_line, xmax); with strenght k=[1,-1], xmin,xmax, xsmooth=2.'''
    def __init__(self, xmin, xmax, strength=[1.0,-1.0], xsmooth=2):
        self.smooth_xmin = Logistic(strength=strength[0], slope=xsmooth, displacement=xmin)
        self.smooth_xmax = Logistic(strength=strength[1], slope=xsmooth, displacement=xmax)
     
    def potential(self,x_line):
        logistic_sum=self.smooth_xmin.potential(x_line)+self.smooth_xmax.potential(x_line)
        return logistic_sum/(logistic_sum.max())  

    def gradient(self,x_line, y_line):
        #grad = np.array([self.linear1D_x.gradient(x_line), self.linear1D_y.gradient(y_line)])
        #return grad
        pass
