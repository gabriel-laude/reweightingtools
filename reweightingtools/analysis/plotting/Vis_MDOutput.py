#!/usr/bin/env python
# coding: utf-8

import numpy as np
import matplotlib.pyplot as plt
import scipy.constants 
import mdtraj as md


## plot temperature
def T_Ns(Temperature_array,
         figuresize   = (12, 8),
         linewidth    = 5,
         potLabel     = r'$U(x)$',
         gradLabel    = r'Temperature $T$ in $K$',
         xLabel       = r'integration step "nstep"',
         xLabelSize   = 35,
         xLabelPad    = 20,
         yLabel       = r'Temperature $T$ in $K$',
         yLabelSize   = 35,
         yLabelPad    = 20,
         ticksSize    = 25,
         ticksPad     = 5,
         legendArgs   = {'size': 30},
         addAX        = None
        ):
    
    xLimits      = [0, len(Temperature_array)]
    yLimits      = None  #[0, 1]
    
    fig, ax = plt.subplots(figsize=figuresize)
    ax.plot(Temperature_array, linewidth=linewidth, label= potLabel)
    if type(addAX) is np.ndarray:
        ax.plot(addAX, linewidth=linewidth, label= potLabel)

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)

    #ax.legend(prop=legendArgs)
    fig.tight_layout()
    #plt.savefig('./Fig_Temperature:nstep.png' , bbox_inches='tight', format='png')

## plot energies
def E_Ns(E_kin, E_pot,
         figuresize   = (12, 8),
         linewidth    = 5,
         potLabel     = r'$U(x)$',
         gradLabel    = r'Temperature $T$ in $K$',
         xLabel       = r'integration step "nstep"',
         xLabelSize   = 35,
         xLabelPad    = 20,
         yLabel       = '$E_x$, x=$\{\mathrm{pot, kin, total}\}$ \n in kJ/mol',
         yLabelSize   = 35,
         yLabelPad    = 20,
         ticksSize    = 25,
         ticksPad     = 5,
         legendArgs   = {'size': 30}
        ):
    
    xLimits      = [0, len(E_kin)]
    yLimits      = None
    
    fig, ax = plt.subplots(figsize=figuresize)
    ax.plot(E_kin, linewidth=linewidth, label= r'$E_\mathrm{kin}$')
    ax.plot(E_pot, linewidth=linewidth, label= r'$E_\mathrm{pot}$')
    ax.plot(E_kin+E_pot, linewidth=linewidth, label= r'$E_\mathrm{total}$')

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    ax.set_xlim(xLimits)
    ax.set_ylim(yLimits)

    ax.legend(prop=legendArgs)
    fig.tight_layout()
    #plt.savefig('./Fig_Energies:nstep.png' , bbox_inches='tight', format='png')


## plot distances 
def Dis_Ns(distances, nstxout, dt,
           figuresize   = (12, 8),
           linewidth    = 5,
           potLabel     = r'$U(x)$',
           gradLabel    = r'Dintaces in nm',
           xLabel       = r'Integration Time in fs',
           xLabelSize   = 35,
           xLabelPad    = 20,
           yLabel       = r'Distance in nm',
           yLabelSize   = 35,
           yLabelPad    = 20,
           ticksSize    = 25,
           ticksPad     = 5,
           legendArgs   = {'size': 30},
           esc_limit_p  = 0.47,
           esc_limit_n  = 0.35,
           esc_limit_m  = 0.40,
           esc_limit    = False,
           addAX        = None
          ):
    
    xLimits      = [0, len(distances)]
    yLimits      = None  #[0, 1],

    fig, ax = plt.subplots(figsize=figuresize)

    t = np.arange(distances.size)
    t = t*nstxout*dt 
    ax.plot(t,distances, linewidth=linewidth, label= r'first')
    if esc_limit:
            plt.hlines(esc_limit_p, 0., len(distances)*nstxout*dt , lw=2)
            plt.hlines(esc_limit_n, 0., len(distances)*nstxout*dt , lw=2)
            plt.hlines(esc_limit_m, 0., len(distances)*nstxout*dt , lw=2)
            
    if type(addAX) is np.ndarray :
        t = np.arange(addAX.size)
        t = t*nstxout*dt 
        ax.plot(t,addAX, linewidth=linewidth, label= r'added')
        

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    #ax.set_xlim(xLimits)
    #ax.set_ylim(yLimits)

    #ax.legend(prop=legendArgs)
    fig.tight_layout()
    #plt.savefig('./Fig_Distances:time.png' , bbox_inches='tight', format='png')



## plot Free Energy 
def His_FreeE_Dis(distances, T, cutoff,
                  figuresize   = (12, 8),
                  linewidth    = 5,
                  potLabel     = r'$U(x)$',
                  gradLabel    = r'Free Energy in kJ/mol',
                  xLabel       = r'Distance in nm',
                  xLabelSize   = 35,
                  xLabelPad    = 20,
                  yLabel       = r'Free Energy in kJ/mol',
                  yLabelSize   = 35,
                  yLabelPad    = 20,
                  ticksSize    = 25,
                  ticksPad     = 5,
                  legendArgs   = {'size': 30},
                  addAX        = None
                 ):

    xLimits      = [min(distances), cutoff]
    yLimits      = None  #[0, 1],
            
    hist, bin_edges = np.histogram(distances, bins=200, density=True)
    bin_array   = bin_edges[:-1] + (bin_edges[-1] - bin_edges[-2])/2

    fig, ax = plt.subplots(figsize=figuresize)
    ax.plot(bin_array,hist, linewidth=linewidth, label= r'Free Energy in kJ/mol')

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel('Histogram', fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    #ax.set_xlim(xLimits)
    #ax.set_ylim(yLimits)
    #plt.savefig('./Fig_NumberOfStates:distances.png' , bbox_inches='tight', format='png')

    fig, ax = plt.subplots(figsize=figuresize)
    free_E = (-T*scipy.constants.Boltzmann*np.log(hist))*(10**(-3)*scipy.constants.Avogadro)
    ax.plot(bin_array,free_E, linewidth=linewidth, label= r'Free Energy in kJ/mol')
    if type(addAX) is np.ndarray: 
        hist, bin_edges = np.histogram(addAX, bins=200, density=True)
        bin_array   = bin_edges[:-1] + (bin_edges[-1] - bin_edges[-2])/2
        free_E = (-T*scipy.constants.Boltzmann*np.log(hist))*(10**(-3)*scipy.constants.Avogadro)
        ax.plot(bin_array,free_E, linewidth=linewidth, label= r'Free Energy in kJ/mol')

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    #ax.set_xlim(xLimits)
    #ax.set_ylim(yLimits)

    #ax.legend(prop=legendArgs)
    fig.tight_layout()
    #plt.savefig('./Fig_FreeEnergy:distances.png' , bbox_inches='tight', format='png')

def Mtp_DisNs(Distances,
              NumberRuns,
              figuresize   = (12, 8),
              linewidth    = 1.5,
              potLabel     = r'$U(x)$',
              gradLabel    = r'Dintaces in nm',
              xLabel       = r'Integration Step',
              xLabelSize   = 35,
              xLabelPad    = 20,
              yLabel       = r'Distance in nm',
              yLabelSize   = 35,
              yLabelPad    = 20,
              ticksSize    = 25,
              ticksPad     = 5,
              legendArgs   = {'size': 30},
              esc_limit_p  = 0.,
              esc_limit_n  = 0.,
              esc_limit_m  = 0,
              esc_limit    = False,
             ):
    
    #xLimits      = [0, len(distances)]
    #yLimits      = None  #[0, 1],

    fig, ax = plt.subplots(figsize=figuresize)

    for run_i in range(NumberRuns):
        
        y = Distances[run_i]
        End = len(y)
        x = np.array(range(End)) 
        ax.plot(x, y, linewidth=linewidth, label= r'Run'+str(run_i))
    if esc_limit:
        plt.hlines(esc_limit_p, 0., End, lw=2)
        plt.hlines(esc_limit_n, 0., End, lw=2)
        plt.hlines(esc_limit_m, 0., End, lw=2)

    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    #ax.set_xlim(xLimits)
    #ax.set_ylim(yLimits)

    ax.legend(prop=legendArgs,bbox_to_anchor=(1,1), loc="upper left")
    fig.tight_layout()
    #plt.savefig('./Fig_Distances:time.png' , bbox_inches='tight', format='png')

    
def Mtp_FreeEDis(List_binArray, List_potEnergySurface, 
                 esc_condition , min_s, min_l,
                 figuresize   = (12, 8),
                 linewidth    = 5,
                 potLabel     = r'$U(x)$',
                 gradLabel    = r'Free Energy in kJ/mol',
                 xLabel       = r'Distance in nm',
                 xLabelSize   = 35,
                 xLabelPad    = 20,
                 yLabel       = r'Free Energy in kJ/mol',
                 yLabelSize   = 35,
                 yLabelPad    = 20,
                 ticksSize    = 25,
                 ticksPad     = 5,
                 legendArgs   = {'size': 30},
                ):
    
    
    ## write extra function
    fig, ax = plt.subplots(figsize=figuresize)
    for graph in range(len(List_binArray)):
        ax.plot(List_binArray[graph], List_potEnergySurface[graph], linewidth=linewidth, label= r'Free Energy in kJ/mol')
    plt.vlines(esc_condition, min(List_potEnergySurface[graph]), max(List_potEnergySurface[graph]) , lw=2)
    plt.scatter(min_s[0],min_s[1])
    plt.scatter(min_l[0],min_l[1])
   
    ax.set_xlabel(xLabel, fontsize=xLabelSize, labelpad=xLabelPad)
    ax.set_ylabel(yLabel, fontsize=yLabelSize, labelpad=yLabelPad)
    ax.tick_params(axis='both', labelsize=ticksSize, pad=ticksPad)
    #ax.set_xlim(xLimits)
    #ax.set_ylim(yLimits)
    


#####


def get_distances(dcd_output, strc_input, 
                  NumberAtom_1, NumberAtom_2, 
                  nstxout, dt, 
                  periodic=True):
    trj = md.load(dcd_output, top = strc_input)
    #print('Selected atoms in the box: '+str(Ca)+', '+str(Cl))
    distances  = md.compute_distances(trj,[[NumberAtom_1,NumberAtom_2]],periodic=periodic) 
    #print('finished')
    
    t = np.arange(distances.size)
    t = t*nstxout*dt #._value 
    
    return t,distances[:,0]    ## achtung das ist anders 


def plot_PoissonCeck(esc_times, ECDF, CDF, tau):
    fig, ax = plt.subplots(figsize=(12,8))
    
    x = np.sort(esc_times)
    y = ECDF
    ax.scatter(x,y, marker='o')
    
    t = np.arange(0,max(x)+10*max(x))
    ax.plot(t,CDF(t,tau))
    
    ax.set_xlabel(r'Escape Time $t_{esc}$ in fs', fontsize=23)
    ax.set_ylabel(r'Escape Probability', fontsize=23)
    ax.tick_params(axis='both', which='major', labelsize=18)
    plt.xscale('log')
    #ax.legend(fontsize=22)
    #return x,y
    