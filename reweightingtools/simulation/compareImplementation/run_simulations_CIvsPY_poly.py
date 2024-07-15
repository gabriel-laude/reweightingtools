''' compare 100 step trajectory of different implementation and different bias types
Mueller Brown potential with linear and polynomial bias
'''
#%% import reweightingtools
import numpy as np
from datetime import datetime as dati
import os 
import sys
try:
    dir_path = os.path.dirname(os.path.realpath(__file__))
except:
    dir_path = os.getcwd()
dir_path = os.path.join(dir_path, os.pardir) 
module_dir = os.path.abspath(dir_path)
sys.path.insert(0, module_dir)

from reweightingtools.integration.integrators import ABOBA
from reweightingtools.potential.MBP import mmMueller, PolynomialBias, pyMueller
from reweightingtools.potential.bias2D import Polynomial
from reweightingtools.analysis.timeseries.reweightingfactors import M_ABOBA

from openmm.unit import nanometer, kelvin, picoseconds, femtoseconds, kilojoules_per_mole
from openmm import CustomIntegrator, System, Context
from openMM_CI_functions import global_variable_names, addComputeTemperatureDependentConstants


#%% import reweightingtools
output_directory = './openMM_MBP_poly/'
nsteps= 101 
nstxout=1
T = 300
kB=0.008314
xi = 5
step = 0.5
nParticles=1
mass=1
temperature=T*kelvin
collision_rate=xi*(picoseconds)**(-1)
timestep=step*femtoseconds
platform='CPU'
constraint_tolerance=1e-8
startingPositions= (np.random.rand(nParticles, 3) * np.array([-0.5,1.5,1])) 
analyticforce = mmMueller()
perturbation = PolynomialBias()

#%% openMM custom implementation    
# logging    
log = open(output_directory + "/MD.log", 'a')
log.write("Simulation Start " + str(dati.now()) + "\n" )
log.write('potential        :  ' + str(analyticforce.__class__) + " \n")
log.write('perturbation     :  ' + str(perturbation.__dict__) + " \n")
log.write('nsteps           :  ' + str(nsteps) + " \n")
log.write('nstxout          :  ' + str(nstxout) + " \n")
log.write('temperature      :  ' + str(temperature) + " \n")
log.write('collision_rate   :  ' + str(collision_rate) + " \n")
log.write('timestep         :  ' + str(timestep) + " \n")
log.close();    

# integrator constants
gamma=collision_rate 
dt=timestep
h=timestep/1

# initialize custom integrator like -> ThermostatedIntegrator -> LangevinIntegrator -> LangevinSplittingGirsanov
Integrator = CustomIntegrator(dt)

Integrator.addGlobalVariable("a", np.exp(-gamma * h))
Integrator.addGlobalVariable("b", np.sqrt(1 - np.exp(- 2 * gamma * h)))
Integrator.addPerDofVariable("x1", 0)
# Set constraint tolerance
Integrator.setConstraintTolerance(constraint_tolerance)
# Add global variables
Integrator.addPerDofVariable("sigma", 0)
Integrator.addGlobalVariable("n", 0)
Integrator.addGlobalVariable("timestep", dt)
Integrator.addGlobalVariable("tau", nstxout)      
Integrator.addGlobalVariable("ndivtau", 0)
Integrator.addGlobalVariable("onedelta", 0)     
Integrator.addGlobalVariable("SumOverPath", 0)
Integrator.addGlobalVariable("M", 0)   
Integrator.addPerDofVariable("Eta0",0)
Integrator.addPerDofVariable("DeltaEta0",0)
Integrator.addPerDofVariable("ff0",0)
Integrator.addPerDofVariable("f_all",0)
Integrator.addComputePerDof("f_all", "f")

Integrator.addUpdateContextState()
Integrator.addGlobalVariable('kT', kB * temperature) 
addComputeTemperatureDependentConstants(Integrator, {"sigma": "sqrt(kT/m)"}) 

# "R V O V R" : (1, ['1/(b*sigma*m) * (1 + a) * timestep/2 * ff0'], "R U V O V R")
# Add random number
Integrator.addComputePerDof("Eta0","gaussian")
# update positions 
Integrator.addComputePerDof("x", "x + ((dt / 2) * v)")
Integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
Integrator.addConstrainPositions()  # x is now constrained
Integrator.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
Integrator.addConstrainVelocities()
# Update forces
Integrator.addComputePerDof("ff0","f1")  
# update delta eta 
Integrator.addComputePerDof("DeltaEta0", '1/(b*sigma*m) * (1 + a) * timestep/2 * ff0')      
# update velocities
Integrator.addComputePerDof("v", "v + ((dt / 2) * f / m)")
Integrator.addComputePerDof("f_all", "f")
Integrator.addConstrainVelocities()
# O_step
Integrator.addComputePerDof("v", "(a * v) + (b * sigma * Eta0)")
Integrator.addConstrainVelocities()
# update velocities
Integrator.addComputePerDof("v", "v + ((dt / 2) * f / m)")
Integrator.addConstrainVelocities()
# update positions 
Integrator.addComputePerDof("x", "x + ((dt / 2) * v)")
Integrator.addComputePerDof("x1", "x")  # save pre-constraint positions in x1
Integrator.addConstrainPositions()  # x is now constrained
Integrator.addComputePerDof("v", "v + ((x - x1) / (dt / 2))")
Integrator.addConstrainVelocities()

Integrator.addComputeGlobal("ndivtau", "n/tau")
Integrator.addComputeGlobal("onedelta","1 - delta(ndivtau-floor(ndivtau))") 
SOP = str()
SOP+="Eta0 * DeltaEta0 + 0.5 * (DeltaEta0 * DeltaEta0)"
Integrator.addComputeSum("SumOverPath", SOP) 
Integrator.addComputeGlobal('M', "M * onedelta + SumOverPath")
Integrator.addComputeGlobal("n", "n + 1")

# define system, analytic force, bias, initial position
integrator=Integrator
system = System()
for i in range(nParticles):
    system.addParticle(mass)
    analyticforce.addParticle(i, [])
    perturbation.addParticle(i, [])
system.addForce(analyticforce)
perturbation.setForceGroup(1) # hard coded
system.addForce(perturbation)
context = Context(system, integrator)
context.setPositions(startingPositions)
context.setVelocitiesToTemperature(temperature)

startingVelocity = context.getState(getVelocities=True).getVelocities(asNumpy=True)
startingVelocity = np.array(startingVelocity)

# write out simulation data 
Mvalues=[]
gvalues=[]
etavalues=[]
DeltaEtavalues=[]
positions=[]
velocities=[]
f_all=[]
bias=[]
trajlength=int(nsteps/nstxout)
for i in range(trajlength):
    position = context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
    positions.append(position)
    velocity = context.getState(getVelocities=True).getVelocities(asNumpy=True)#.value_in_unit(nanometer_per_picosecond)
    velocities.append(velocity)
    etavalues.append(integrator.getPerDofVariableByName("Eta0"))
    DeltaEtavalues.append(integrator.getPerDofVariableByName("DeltaEta0"))
    f_all.append(integrator.getPerDofVariableByName("f_all"))
    bias.append(integrator.getPerDofVariableByName("ff0"))
    Mvalues.append(integrator.getGlobalVariableByName("M"))
    state  = context.getState(getEnergy = True, groups = 0b00000000000000000000000000000010) 
    energy = state.getPotentialEnergy().value_in_unit(kilojoules_per_mole) #unit.kilojoules_per_mole)
    gvalues.append(energy)   
    integrator.step(nstxout)
           
## save trajectory
np.save(output_directory+'positions', np.squeeze(np.array(positions), axis=1))
np.save(output_directory+'velocities', np.squeeze(np.array(velocities), axis=1))
np.save(output_directory+'total_force', np.squeeze(np.array(f_all), axis=1))
np.save(output_directory+'eta', np.squeeze(np.array(etavalues), axis=1))
values = np.array(np.stack([Mvalues,gvalues]).swapaxes(0,1))
np.savetxt(output_directory+'ReweightingFactors.txt',values)
np.save(output_directory+'DeltaEta', np.squeeze(np.array(DeltaEtavalues), axis=1))
np.save(output_directory+'bias_force', np.squeeze(np.array(bias), axis=1))
   
#%% python implementation
eta = np.load(output_directory+'eta.npy')
eta = eta[:,:2]
output_directory='./python_MBP_poly/'
poly_x = Polynomial(strength=50)
mb=pyMueller(bias=poly_x)
q_initial = startingPositions[0][:2]
p_initial = startingVelocity[0][:2]

qABOBA, pABOBA, etaABOBA, fABOBA = ABOBA(potential_gradient=mb.gradient, 
                                              position_inital= q_initial, 
                                              momentum_inital= p_initial,
                                              n_steps=nsteps-1, 
                                              time_step=step*0.001, 
                                              m=mass*np.ones_like(q_initial), 
                                              xi=xi, 
                                              kB=kB, 
                                              T=T, 
                                              eta= eta) 

bias_force_py = -poly_x.gradient(qABOBA[:,0], qABOBA[:,1]).swapaxes(0,1)
bias_py = poly_x.potential(qABOBA[:,0], qABOBA[:,1]).swapaxes(0,1)

print(bias_force_py.shape)
print(bias_py.shape)

deltaEta, M_x = M_ABOBA(bias_force_py[:,0], eta[:,0], step*0.001, T, mass, xi, kB)

np.save(output_directory+'positions', qABOBA)
np.save(output_directory+'velocities', pABOBA)
np.save(output_directory+'total_force', fABOBA)
np.save(output_directory+'eta', etaABOBA)
values = np.array(np.stack([M_x,bias_py[:,0]]).swapaxes(0,1))
np.savetxt(output_directory+'ReweightingFactors.txt',values)
np.save(output_directory+'DeltaEta', deltaEta)
np.save(output_directory+'bias_force', bias_force_py)

#%% visualisation of diff 
#output_directory='./openMM_MBP_5z/'
#quant = np.loadtxt(output_directory+'ReweightingFactors.txt')
#output_directory='./python_MBP_5z/'
#quant2 = np.loadtxt(output_directory+'ReweightingFactors.txt')
#import reweightingtools.analysis.plotting.visualizations as vis
#x = np.arange(len(quant[:,0]))
#vis.plot_quantity([x,x],[quant[:,1],quant2[:,1]],['C1','C2'])