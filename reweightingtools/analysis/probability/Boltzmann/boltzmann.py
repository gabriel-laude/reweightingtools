#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Feb 17 16:18:05 2023

@author: schaefej51

This file containes function to create Hisrogramms of 1D and 2D MD data.
"""
import numpy as np
from scipy import integrate

import os
import sys
# Add the directory containing your module to the Python path (before importing it)
module_dir = os.path.abspath('/home/schaefej51/Documents/2_Projects/reweightingtools')
sys.path.insert(0, module_dir)
# Now you can import your module as usual
from reweightingtools.analysis.plotting.visualizations import plot_BoltzmannCeck, plot_quantity, plot_histogram2D


# sampled Boltzmann
# histogram
def histogram1D(traj1D, nbins, density=True, normed=True):
    '''gives 1D histogram with centralized edges'''
    hist, bins = np.histogram(traj1D, nbins, density=density)
    edges = bins[:-1] + 0.5 * (bins[-1] - bins[-2])
    if normed:
        hist/=hist.max()
    return hist, edges

def histogram2D(traj2D, nbinsx, nbinsy, density=True, normed=False):
    '''gives 2D histogram with centralized edges and bin arrays'''
    x, y = traj2D[:,0], traj2D[:,1]
    hist, binsx, binsy = np.histogram2d(x, y, bins=(nbinsx, nbinsy), density=density)
    xedges=binsx[:-1] + 0.5 * (binsx[1:] - binsx[:-1])
    yedges=binsy[:-1] + 0.5 * (binsy[1:] - binsy[:-1])
    hist=hist.T
    if normed:
        hist/=hist.max()
    return hist, xedges, yedges, binsx, binsy

def analytic_Boltzmann1D(potential, edges, beta, normed=False):
    '''gives 1D analytical Boltzmann distribution'''
    boltzmann = np.exp(-beta * potential(edges))
    norm = integrate.trapz(boltzmann, edges)
    delta = edges[1] - edges[0]
    boltzmann = (delta/norm) * np.exp(-beta * potential(edges))
    if normed:
        return boltzmann / boltzmann.max() 
    else:
        return boltzmann

def check_Boltzmann1D(traj1D, potential, nbins=100, beta=1.0, density=True, normed=True, absolute=True, relative=True, both=True):
    '''gives 1D analytical Boltzmann distribution'''
    samp, edges = histogram1D(traj1D, nbins, density=density, normed=normed)
    ana = analytic_Boltzmann1D(potential, edges, beta=beta, normed=normed) 
    diff=samp-ana
    if relative:
        ana_n = np.where(ana == 0, ana, 0.00000000001)
        diff/=ana
        plot_quantity(edges,diff, color='b', xlabel=r'x', ylabel=r'distribution relative difference', label=None)
    if absolute:
        plot_quantity(edges,diff, color='b', xlabel=r'x', ylabel=r'distribution absolute difference', label=None)
    if both:
        plot_BoltzmannCeck(edges,samp,ana)


def analytic_Boltzmann2D(potential, xedges, yedges, beta, normed=False):
    '''gives 2D analytical Boltzmann distribution'''
    xy =  np.meshgrid(xedges,yedges)
    Boltzmann    = lambda x, y : np.exp(-beta * potential(x, y))        
    norm, error = integrate.dblquad(Boltzmann, min(xedges), max(xedges), min(yedges), max(yedges))
    delta  = (xy[0][0][1] - xy[0][0][0]) * (xy[1][1][1] - xy[1][0][0])
    boltzmann = delta/norm * np.exp(-beta * potential(xy[0], xy[1]))
    if normed:
        return boltzmann / boltzmann.max() 
    else:
        return boltzmann


def check_Boltzmann2D(traj2D, potential, nbinsx=100, nbinsy=100, beta=1.0, density=True, normed=True, absolute=True, relative=True, clip=False, **kwargs):
    '''gives 1D analytical Boltzmann distribution'''
    samp, xedges, yedges, _, _ = histogram2D(traj2D, nbinsx, nbinsy, density=density, normed=normed)
    ana = analytic_Boltzmann2D(potential, xedges, yedges, beta, normed=normed) 
    diff = samp-ana
    if absolute:
        plot_histogram2D(xedges, yedges, diff, title=None,xlabel=r'x',ylabel=r'y',
                         colorbartitle='absolute density difference', **kwargs)
    if relative:
        ana_n = np.where(ana == 0, ana, 0.00000000001)
        diff /=ana_n
        if clip:
            diff = diff.clip(max=1, min=-1)
        plot_histogram2D(xedges,yedges,diff,title=None,xlabel=r'x',ylabel=r'y',
                         colorbartitle='relative density difference', **kwargs)
