#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Implementation of Müller Brown potential for openMM simulations. 
@author: 2020 Robert T. McGibbon
added: J.-L. Schaefer 2023
"""
import numpy as np
import matplotlib.pyplot as plt
from openmm import CustomExternalForce

# openmm potential classes 
class mmMueller(CustomExternalForce):
    """OpenMM custom force for propagation on the Muller Potential. Also
    includes pure python evaluation of the potential energy surface so that
    you can do some plotting"""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-20, -10, -17, 1.5]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]

    def __init__(self):
        # start with a harmonic restraint on the Z coordinate
        expression = '1000.0 * z^2'
        for j in range(4):
            # add the muller terms for the X and Y
            fmt = dict(aa=self.aa[j], bb=self.bb[j], cc=self.cc[j], AA=self.AA[j], XX=self.XX[j], YY=self.YY[j])
            expression += '''+ {AA}*exp({aa} *(x - {XX})^2 + {bb} * (x - {XX}) 
                               * (y - {YY}) + {cc} * (y - {YY})^2)'''.format(**fmt)
        
        super(mmMueller, self).__init__(expression)
    
    @classmethod
    def potential(cls, x, y, bias=None):
        '''Compute the potential at a given point x,y. May include a bias. However,
        for openMM biased potential use custom external force to bias potential in __init__.'''
        value = 0
        for j in range(4):
            value += cls.AA[j] * np.exp(cls.aa[j] * (x - cls.XX[j])**2 + \
                cls.bb[j] * (x - cls.XX[j]) * (y - cls.YY[j]) + cls.cc[j] * (y - cls.YY[j])**2) 
        if bias is not None:
            value+=bias(x,y)
        return value

    @classmethod
    def plot(cls, ax=None, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, bias=None, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 100.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = cls.potential(xx, yy, bias)
        # clip off any values greater than 200, since they mess up
        # the color scheme
        fig, ax = plt.subplots()
        if ax is None:
            ax = plt
        im=ax.contourf(xx, yy, V.clip(min=-15,max=5), 40, **kwargs)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        cbar = plt.colorbar(im)
        cbar.set_label('potential in kJ/mol',fontsize=13)
        ax.tick_params(labelsize=13)
        plt.show()

        
class LinearBias(CustomExternalForce):
    """OpenMM custom linear bias force for propagation on the Muller Potential."""
    def __init__(self, strength):
        fmt = dict(k=strength)
        expression = '''{k} * x''' .format(**fmt)
        
        super(LinearBias, self).__init__(expression) #,strength 

        
class PolynomialBias(CustomExternalForce):
    """OpenMM custom polynomial bias force for propagation on the Muller Potential."""
    aa = [-1, -1, -6.5, 0.7]
    bb = [0, 0, 11, 0.6]
    cc = [-10, -10, -6.5, 0.7]
    AA = [-20, -10, -17, 1.5]
    XX = [1, 0, -0.5, -1]
    YY = [0, 0.5, 1.5, 1]        
    
    def __init__(self):
        fmt = dict(a = 2.05777925e-02, b = (-2.31737460e-02), c = 3.83236939e-03 , 
                   d = 3.91666334e-02, e = (-1.39425035e-02), f = (-3.38910809e-02), 
                   g = 3.81706681e-04, h = 1.24385771e-02, i = 3.36985889e-03, 
                   j = (-1.17944563e-03), k = (-7.50078923e-04), 
                   l = (-1.37745928e-04), m =  (-0.00000881084849))
        expression = '''-50 *({a} + {b} * x + {c} * x ^ 2 + {d} * x ^ 3 + {e} * x ^ 4 + {f} * x ^ 5 + {g} * x ^ 6 + {h} * x ^ 7+ {i} * x ^ 8 + {j} * x ^ 9+{k} * x ^ 10 + {l} * x ^ 11+ {m} * x ^ 12)'''.format(**fmt)
        
        super(PolynomialBias, self).__init__(expression)

        
# python reference implementation
class pyMueller():
    """Python Mueller Brown potential for python reference simulation.
    Includes potential energy surface and its gradienr, optional add a bias.
    ref: 
    
    inpit bias func : bias function of 2 input variables.
    """
   
    def __init__(self, bias): #, **para): 
        #self.para = para 
        self.bias = bias
        
        # hard coded
        self.A_n = np.array([[-20,-10],[-17,1.5]])
        self.a_n = np.array([[-1,-1],[-6.5,0.7]])
        self.b_n = np.array([[0,0],[11,0.6]])
        self.c_n = np.array([[-10,-10],[-6.5,0.7]])
        self.x_n = np.array([[1,0],[-0.5,-1]])
        self.y_n = np.array([[0.,0.5],[1.5,1]])
        self.N   = np.arange(self.A_n.shape[0])
        
        super(pyMueller, self).__init__()
    

    def summand(self, l, n, x, y):
        return self.A_n[l,n] * np.exp(self.a_n[l,n] * (x - self.x_n[l,n])**2 +\
                                 self.b_n[l,n] * (x - self.x_n[l,n]) * (y - self.y_n[l,n]) +\
                                         self.c_n[l,n] * (y - self.y_n[l,n])**2)

    def mueller_brown(self, l, x, y):
        return sum(pyMueller.summand(self, l, n, x, y) for n in self.N)
    
    def v_prime_X(self, l, n, x, y):
        return 2 * self.a_n[l,n] * (x - self.x_n[l,n]) +\
                self.b_n[l,n] * (y - self.y_n[l,n])
    
    def mueller_brown_gradient_X(self, l, x, y):
        return sum(pyMueller.v_prime_X(self, l, n, x, y) *\
                   pyMueller.summand(self, l, n, x, y) for n in self.N)
    
    def v_prime_Y(self, l, n, x, y):
        return self.b_n[l,n] * (x - self.x_n[l,n]) +\
                2 * self.c_n[l,n] * (y - self.y_n[l,n])
            
    def mueller_brown_gradient_Y(self, l, x, y):
            return sum(pyMueller.v_prime_Y(self,l, n, x, y) *\
                       pyMueller.summand(self, l, n, x, y) for n in self.N)
    
    
    def potential(self, x, y, **kwargs):
        pot = pyMueller.mueller_brown(self, 0, x, y) + pyMueller.mueller_brown(self, 1, x, y)
        if self.bias is not None:
            pot += self.bias.potential(x, y, **kwargs)
        return  pot
    
    def gradient(self, vector_position, **kwargs):
        x, y = vector_position
        grad = np.array([pyMueller.mueller_brown_gradient_X(self, 0, x, y) + pyMueller.mueller_brown_gradient_X(self, 1, x, y),\
             pyMueller.mueller_brown_gradient_Y(self, 0, x, y) + pyMueller.mueller_brown_gradient_Y(self, 1, x, y)])
        if self.bias is not None:
            grad += self.bias.gradient(x, y, **kwargs)
        return grad
    
    def plot(self, minx=-1.5, maxx=1.2, miny=-0.2, maxy=2, **kwargs):
        "Plot the Muller potential"
        grid_width = max(maxx-minx, maxy-miny) / 100.0
        ax = kwargs.pop('ax', None)
        xx, yy = np.mgrid[minx : maxx : grid_width, miny : maxy : grid_width]
        V = self.potential(xx, yy, **kwargs)
        fig, ax = plt.subplots()
        im=ax.contourf(xx, yy, V.clip(min=-15,max=5), 40, **kwargs)
        ax.tick_params(axis='both', which='major', labelsize=13)
        ax.set_xlabel('x', fontsize=13)
        ax.set_ylabel('y', fontsize=13)
        cbar = plt.colorbar(im)
        cbar.set_label('potential in kJ/mol', fontsize=13)
        ax.tick_params(labelsize=13)
        plt.show()
