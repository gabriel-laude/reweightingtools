#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
@author: schaefej51

This file includes functions to read, write or get infos from openMM MD simulations.
"""

import numpy as np
import matplotlib.pyplot as plt
#import matplotlib.gridspec as gridspec

def plot_potential_class(x_line, 
                         potential_class, 
                         figuresize   = (9, 5),  
                         linewidth    = 5,
                         potLabel     = r'$U(x)$',
                         gradLabel    = r'$F=-\nabla_x U(x)$',
                         xLabel       = r'$x$ in a.u.',
                         xLabelSize   = 15,
                         xLabelPad    = 15,
                         yLabel       = r'$U(x), F=(x)$ in Hartree',
                         yLabelSize   = 15,
                         yLabelPad    = 15,
                         ticksSize    = 15,
                         ticksPad     = 5,
                         legendArgs   = {'size': 20},
                         xLimits      = None, #[0, 1],
                         yLimits      = None,  #[0, 1],
                         plotGradient = True
                        ):
    ''' to plot the potential or/and its gradient.
    Arguments:
        x_line: 1D array of x value range.
        potential_class: class according to reweighting.potential.
        plotGradient: boolean if True plot gradient.
    '''
    potential, gradient = potential_class.potential, potential_class.gradient
    
    fig, ax = plt.subplots(figsize=figuresize)
    ax.plot(x_line,potential(x_line), linewidth=linewidth, label= potLabel)
    if plotGradient == True:
        ax.plot(x_line,-gradient(x_line), linewidth=linewidth, label=gradLabel)
                
    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)
            
    ax.legend(prop=legendArgs)
    fig.tight_layout()

def plot_trajectory(trajectory, 
                    timestep,
                    delta=100,
                    figuresize = (6,7),
                    ylabel=r'timesteps in ps',
                    xlabel=r'$x$',
                    grid=True,
                    dots=False,
                    name='traj',
                    save=False): 
    '''To plot 1D trajectory.
    Arguments:
        trajectory: 1D trajectory.
        timestep: integration time step of simulation.
        delta: slicing parameter.
    '''
    length = len(trajectory)*timestep
    intsteps = np.arange(0,length,timestep)

    fig, ax = plt.subplots(figsize=figuresize)
    if dots:
        ax.plot(trajectory[::5*delta], intsteps[::5*delta], '.', color='C0', lw=0.4)
    ax.plot(trajectory[::delta], intsteps[::delta], color='C0', lw=0.4)

    ax.grid(grid, axis='x',color='C7', lw=1)          
    ax.set_ylabel(ylabel, fontsize=15)
    ax.set_xlim((min(trajectory), max(trajectory)))
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel(xlabel, fontsize=15)
    if save:
        plt.savefig(name+'.png' , bbox_inches='tight', format='png')   
    fig.tight_layout()
    plt.show()
        
def plot_BoltzmannCeck(edges,
                       samp,
                       ana,
                       figsize=(9, 5),
                       xlabel=r'position $x$',
                       ylabel=r'Boltzmann distribution',
                       labelsize=13,
                       fontsize=13,
                       lw=1
                      ):
    '''Plot analytic and sampled 1D Boltzmann distribution.
    Argumetns:
        edges: array of centers of bin array.
        ana: analytiv Boltzmann distribution.
        samp: sampled Boltzmann distribution.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(edges, ana, lw=lw, c='black', label=r'analytic')
    ax.plot(edges, samp, '--', lw=lw, c='C1', label=r'sampled')
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.legend(fontsize=fontsize)

def plot_quantity(quantityx,
                  quantityy,
                  color='C1',
                  figsize=(9, 5),
                  xlabel=r'quantityx',
                  ylabel=r'quantityy',
                  label=r'quantity',
                  labelsize=13,
                  fontsize=13,
                  lw=1):
    '''Plot a 1D quantity.
    Arguments:
        quantityx: array for x values.
        quantityy: array for y values.
    '''
    fig, ax = plt.subplots(figsize=figsize)
    if isinstance(quantityx, list):
        for i,(x,y) in enumerate(zip(quantityx, quantityy)):
            ax.plot(x, y, lw=lw, c=color[i], label=label[i])
    else:
        ax.plot(quantityx, quantityy, lw=lw, c=color, label=label)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    ax.legend(fontsize=fontsize)
    plt.show()

def plot_histogram2D(x,
                     y,
                     hist2D,
                     title,
                     xlabel,
                     ylabel,
                     colorbartitle,
                     save=False,
                     fig_directory='./histogram2D',
                     size=10,
                     in2cm=1/2.54,  # centimeters in inches
                     subplotX=1,
                     subplotY=1,
                     figsizeX=20,
                     figsizeY=16,
                     fontsize=23,
                     labelsize=23,
                     labelpad=20,
                     **kwargs):
    font = {'size'   : size}
    plt.rc('font', **font)
    '''Plot 2D histogram.
    Arguments:
        x: x values 
        y: y values
        hist2D: 2D array for histogram
    '''
    fig, ax = plt.subplots(subplotX, subplotY, figsize=(figsizeX*in2cm, figsizeY*in2cm))  
    
    pos = ax.pcolor(x, y, hist2D, **kwargs)

    ax.set_title(title, fontsize=fontsize)
    ax.set_xlabel(xlabel, fontsize=fontsize)
    ax.set_ylabel(ylabel, fontsize=fontsize)
    ax.tick_params(axis='both', which='major', labelsize=labelsize)
    cbar = fig.colorbar(pos, ax=ax)
    cbar.ax.tick_params(labelsize=labelsize) 
    cbar.set_label(colorbartitle, fontsize=fontsize, labelpad=labelpad)
    
    if save:
        fig.savefig(fig_directory +'.png' , bbox_inches='tight', format='png')
