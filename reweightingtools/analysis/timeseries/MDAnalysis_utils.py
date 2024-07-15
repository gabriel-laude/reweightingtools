import numpy as np
from datetime import datetime as dati
import matplotlib.pyplot as plt
#from Vis_MDOutput import plot_PoissonCeck

## save and load data 
## ToDo. make nicer
def _get_filename_timesXdistances(run_i, NumberAtom_1, NumberAtom_2, name):
    if name=='distance':
        return 'distances_'+str(run_i)+'.txt'
    elif name=='COLVAR_d':
        return 'COLVAR_d_'+str(run_i)
    elif name=='COLVAR_B_Cl_None':
        return 'COLVAR_B_Cl_None_'+str(run_i)
    elif name=='COLVAR_B_SO4_None':
        return 'COLVAR_B_SO4_None_'+str(run_i)
    elif name=='COLVAR_B_PO4_None':
        return 'COLVAR_B_PO4_None_'+str(run_i)
    elif name=='dist_NA1:NA2':
        return 'dists_'+str(NumberAtom_1)+':'+str(NumberAtom_2)+'_'+str(run_i)+'.txt'
    
def _get_data_timesXdistances(MD_output_directory, filename_distances):
    read_data    = np.loadtxt(MD_output_directory+filename_distances)
    return read_data[:,0], read_data[:,1]

def makeDataContainer_distances(NumberRuns,
                                MD_output_directory,
                                NumberAtom_1, 
                                NumberAtom_2, 
                                old_naming
                               ):
    ''' Function to create Lists containing the time and distance between two atoms at each integration step. 
    '''
    Distances = []
    Simulationtimes = []
    for run_i in range(NumberRuns):
        filename_distances = _get_filename_timesXdistances(run_i, NumberAtom_1, NumberAtom_2, old_naming)
        _, distances   = _get_data_timesXdistances(MD_output_directory, filename_distances)
        
        Distances.append(distances)
        Simulationtimes.append(len(distances))
    return Distances, Simulationtimes

## escape rate utilities

def _check_ESCtraj(distances, esc_condition):
    for nstep, distance in enumerate(distances):
        if distance < esc_condition:
            esc_timestep = False
            continue
            
        else:
            esc_timestep = nstep+1
            break
    return esc_timestep

def check_ESCtraj(Distances, NumberRuns, esc_condition):
    finishedRuns  = []
    restartRuns   = []
    esc_timesteps     = []
    for run_i in range(NumberRuns):
        esc_timestep = _check_ESCtraj(Distances[run_i], esc_condition)
        if esc_timestep != False:
            finishedRuns.append(run_i)
            esc_timesteps.append(esc_timestep)
        else:
            restartRuns.append(run_i)
    return (finishedRuns, esc_timesteps), restartRuns

def get_esc_times(Distances, NumberRuns, esc_condition, dt, nstxout):
    esc_timesteps     = []
    for run_i in range(NumberRuns):
        esc_timestep = _check_ESCtraj(Distances[run_i], esc_condition)
        esc_timesteps.append(esc_timestep)
    return esc_timesteps * dt._value * nstxout

def _esc_rate(esc_times):
    if type(esc_times) is list:
        esc_times = np.array(esc_times)
    return 1/(esc_times.mean())


## Histogram and free energy analysis         
def _get_histogram_distances(distances, cutoff, bins, density):
    hist, bin_edges = np.histogram(distances, bins=bins, density=density)
    bin_array       = bin_edges[:-1] + (bin_edges[-1] - bin_edges[-2])/2
    hist_zeros = np.where(hist==0)[0]
    try:
        missing_microstates = bin_array[hist_zeros]
        if hist_zeros.size != 0:
            for missing_microstate in missing_microstates:
                if missing_microstate < cutoff:
                    print('ATTENTION:\n Not all microstates are sampled! Here microstate %s of %s microstates is missing.' %(missing_microstate, bins))
            hist[hist_zeros]+=1  
    except:
        pass
    
    if cutoff != 0:
        hist            = hist[np.where(bin_array<cutoff)]
        bin_array       = bin_array[np.where(bin_array<cutoff)]
    
    return hist, bin_array

def get_density_Distances(Distances, cutoff, bins, density):
    distances = np.concatenate(Distances)
    hist, bin_array = _get_histogram_distances(distances, cutoff, bins, density)
    return _get_histogram_distances(distances, cutoff, bins, density)   

def get_potentialEnegry(T, density, unit):
    kB  = scipy.constants.Boltzmann
    if unit=='kJ/mol':
        NA  = scipy.constants.Avogadro
        return (-T*kB*np.log(density))*(10**(-3)*NA)
    if unit=='J':
        return (-T*kB*np.log(density))
        
## check data and write log file 
def load_LOG(LOGDirectory):
    ## Load Log-File
    ###########################
    with open(LOGDirectory) as file:
        resources = {}
        units     = {}
        for line in file:
            key, value, unit   = line.rstrip().split(' ', 2)
            resources[key] = value
            units[key]         = unit
        
    nstxout = float(resources['nstxout'])
    dt      = float(resources['dt'])  ##dt._value
    T       = float(resources['T'])
    cutoff  = float(resources['cutoff'])
    
    return nstxout, dt, T, cutoff

def AYSLOG_checkInput(args_filename,
                      NumberRuns,
                      NumberAtom_1_IPF,
                      NumberAtom_2_IPF,
                      nstxout, 
                      dt, 
                      T, 
                      cutoff,
                      Directory,
                      System,
                      Parameter,
                      LOGDirectory,
                      MD_output,
                      AYS_output
                      ):
    
    log = open(AYS_output + "LOG.txt", 'a')
    log.write('                 O U T P U T  A N A L Y S I S \n' )
    log.write('                 created '+ str(dati.now()) + "\n" )
    log.write('                 _____________________________________________________________________\n' )
    log.write('System            :' + str(System) + " \n")
    log.write('Parameter         :' + str(Parameter) + " \n")
    log.write('MD_output         :' + str(MD_output) + " \n")
    log.write('LOGDirectory      :' + str(LOGDirectory) + " \n")
    log.write('AYS_output        :' + str(AYS_output) + " \n")
    log.write('NumberRuns        :' + str(NumberRuns) + " \n")
    log.write('                 _____________________________________________________________________\n' )
    nstxout_out, dt_out, T_out, cutoff_out = load_LOG(LOGDirectory)
    if nstxout_out == nstxout:
        log.write('nstxout      :' + str(nstxout) + " steps \n")
    else:
        log.write('Matching Errors: Input and LOG file have not the same nstxout values.\n') 
        print('Matching Errors: Input and LOG file have not the same nstxout values.\n')
    if dt_out      == dt._value:
        log.write('dt           :' + str(dt) + "\n")
    else:
        log.write('Matching Errors: Input and LOG file have not the same dt values.\n') 
        log.write('dt           :' + str(dt, dt_out) + "\n")
    if T_out       == T._value:
        log.write('T            :' + str(T) + "\n")
    else:
        log.write('Matching Errors: Input and LOG file have not the same T values.\n') 
    if cutoff_out  == cutoff._value:
        log.write("cutoff       :" + str(cutoff) + "\n" )
    else:
        log.write('Matching Errors: Input and LOG file have not the same cutoff values.\n') 
    if NumberAtom_1_IPF  == args_filename['NumberAtom_1']:
        pass
    else:
        log.write('Matching Errors: Input and LOG file have not the same NumberAtom_1 values.\n')
    if NumberAtom_2_IPF  == args_filename['NumberAtom_2']:
        pass
    else:
        log.write('Matching Errors: Input and LOG file have not the same NumberAtom_2 values.\n') 
    log.write('                 _____________________________________________________________________\n' )
    log.close();
    
def AYSLOG_escLimit(AYS_output, args_filename, args_escLimit):
    log = open(AYS_output + "LOG.txt", 'a')
    log.write('args_filename   :\n' + str(args_filename) + "\n")
    log.write('args_escLimit   :\n' + str(args_escLimit) + "\n")
    log.write('                 _____________________________________________________________________\n' )
    log.close();

def AYSLOG_checkSimulationQuality(Distances, 
                                  NumberRuns, 
                                  esc_condition,
                                  dt, 
                                  nstxout,
                                  AYS_output,
                                  inline=False,
                            ):
    log = open(AYS_output + "LOG.txt", 'a')
    log.write('                 _____________________________________________________________________\n' )
    log.write(f' Quality Measures of Trustworthiness of MD Calculations\n')
    log.write('')
    log.write('---------------------------------------------------------\n' )
    finishedRuns, restartRuns = check_ESCtraj(Distances, NumberRuns, esc_condition)
    log.write(' ToDo: Restart MD runs \n %s\n' %(restartRuns))
    if inline:
        print(f' Quality Measures of Trustworthiness of MD Calculations\n')
        print(' ToDo: Restart MD runs %s\n' %(restartRuns))
    esc_timesteps = finishedRuns[1]
    log.write(' Note: Analysis is done for MD runs \n %s\n' %( finishedRuns[0]))
    if inline:
        print(' Note: Analysis is done for MD runs %s\n' %( finishedRuns[0]))
    
    esc_times = esc_timesteps * dt._value * nstxout
    mu,sigma,median,ECDF,tau,rate,sigmaMu,medianMulog2 = check_poisson(esc_times, plot=False, log=True)

    if inline:
        print(f'Measures Sensitive to Insufficient Sampling')
        print(f'sigma/mu    = 1 : {sigma/mu == 1}, {sigma/mu:.5}')
        print(f'tm/(mu*ln2) = 1 : {medianMulog2 == 1}, {medianMulog2:.5}')
        print(f'Estimation of Theoretical Average Rate')
        print(f'mu   = tau_fit  : {mu:.3e} fs   ~ {tau:.3e} fs')
        print(f'rate = 1/tau_fit: {rate:.3e} fs-1 ~ {1/tau:.3e} fs-1')

    log.write(f' according to J.Chem.Theory.Comput. 2014, 10, 1420-1425\n')
    log.write('---------------------------------------------------------\n' )
    log.write(f' Measures Sensitive to Insufficient Sampling\n')
    log.write(f'sigma/mu    = 1 : {sigma/mu == 1}, {sigma/mu:.5}\n')
    log.write(f'tm/(mu*ln2) = 1 : {medianMulog2 == 1}, {medianMulog2:.5}\n')
    log.write(f' Estimation of Theoretical Average Rate\n')
    log.write(f'mu   = tau_fit  : {mu:.3e} fs   ~ {tau:.3e} fs\n')
    log.write(f'rate = 1/tau_fit: {rate:.3e} fs-1 ~ {1/tau:.3e} fs-1\n')
    log.write('                 _____________________________________________________________________\n' )
    log.write('' )
    
    log.close();
