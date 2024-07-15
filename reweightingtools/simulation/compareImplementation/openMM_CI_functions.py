'''This file provides reporter and set up functions for a openMM simulation with a Custom integrator.
'''
from openmm import CustomExternalForce
from openmm.unit import kilojoules_per_mole, nanometer

# functions
def global_variable_names(Int):
    """The set of global variable names defined for this integrator."""
    return set([ Int.getGlobalVariableName(index) for index in range(Int.getNumGlobalVariables()) ])
def addComputeTemperatureDependentConstants(Int, compute_per_dof):
    # First check if flag variable already exist.
    if not 'has_kT_changed' in global_variable_names(Int):
        Int.addGlobalVariable('has_kT_changed', 1)

    # Create if-block that conditionally update the per-DOF variables.
    Int.beginIfBlock('has_kT_changed = 1')
    for variable, expression in compute_per_dof.items():
        Int.addComputePerDof(variable, expression)
    Int.addComputeGlobal('has_kT_changed', '0')
    Int.endBlock()

class PositionReporter(object):
    def __init__(self, file, reportInterval, nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.nParticles=nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        position = simulation.context.getState(getPositions=True).getPositions(asNumpy=True).value_in_unit(nanometer)
        for i in range(self.nParticles):
            print(str(position[i][0]) +' '+str(position[i][1])+' ' +str(position[i][2]), file = self._out)

class VelocityReporter(object):
    def __init__(self, file, reportInterval,nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.nParticles=nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        velocity = simulation.context.getState(getVelocities=True).getVelocities(asNumpy=True)
        for i in range(self.nParticles):
            print(str(velocity._value[i][0]) +' '+str(velocity._value[i][1])+' ' +str(velocity._value[i][2]), file = self._out)

class RandomNumberReporter(object):
    '''
    reporter for ML for momentum space
    '''
    def __init__(self, file, reportInterval,integrator,nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.integrator=integrator
        self.nParticles=nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        etavalues=self.integrator.getPerDofVariableByName("Eta0")
        for i in range(self.nParticles):
            print(str(etavalues[i][0]) +' '+str(etavalues[i][1])+' ' +str(etavalues[i][2]), file = self._out)

class BiasReporter(object):
    '''
    reporter for ML for momentum space
    '''
    def __init__(self, file, reportInterval,integrator,nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.integrator=integrator
        self.nParticles=nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        bias=self.integrator.getPerDofVariableByName("ff0")
        for i in range(self.nParticles):
            print(str(bias[i][0]) +' '+str(bias[i][1])+' ' +str(bias[i][2]), file = self._out)
            
class TotalForceReporter(object):
    '''
    reporter for ML for momentum space
    '''
    def __init__(self, file, reportInterval, integrator,nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.integrator=integrator
        self.nParticles=nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        force=self.integrator.getPerDofVariableByName("f_all")
        for i in range(self.nParticles):
            print(str(force[i][0]) +' '+str(force[i][1])+' ' +str(force[i][2]), file = self._out)

class DeltaEtaReporter(object):
    '''
    reporter for ML for momentum space
    '''
    def __init__(self, file, reportInterval, integrator,nParticles):
        self._out = open(file, 'w')
        self._reportInterval = reportInterval
        self.integrator = integrator
        self.nParticles = nParticles

    def __del__(self):
        self._out.close()

    def describeNextReport(self, simulation):
        steps = self._reportInterval - simulation.currentStep%self._reportInterval
        return (steps, False, False, True, False)

    def report(self, simulation, state):
        DE=self.integrator.getPerDofVariableByName("DeltaEta0")
        for i in range(self.nParticles):
            print(str(DE[i][0]) +' '+str(DE[i][1])+' ' +str(DE[i][2]), file = self._out)
            
def set_restraints(func, system, top, gro, **kwargs):
    if func==None:
        pass
    elif 'multiple_restraints' in list(kwargs.keys()):
        print(kwargs['atom_name'])
        for restraints in range(len(kwargs['atom_name'])):
            rst_kwargs=dict()
            for key, value in kwargs.items():
                print(key, value )
                if key != 'multiple_restraints':
                    rst_kwargs[key]= value[restraints]
            print(rst_kwargs)
            func[restraints](system, top, gro, **rst_kwargs)   
    else:
        print('else')
        return func(system, top, gro, **kwargs)

def restraints_harmonic_force(system, top, gro, **kwargs):
    ''' harmonic force binds each particle to its initial position, by adding a CustomExternalForce that
    set the energy of each particle equals a force_constants multiplied by the square of the periodic 
    distance between the particle’s current position (x, y, z) and a reference position (x0, y0, z0).
    Args:
        system: the OpenMM System object to simulate (or the name of an XML file with a serialized 
        System)
        top: topology stores the topological information about a system.
        gro: constructs a set of atom positions from it, also contains some topological information, such 
        as elements and residue names
        atom_name: tuple of strings refering the topology abbreviation of the atom which should 
        be restrained
        force_constants: float, defines the strength of the replacement force
        restraint_x: (or _y, _z) bool defining if a harmonic force is defined in this direction
    Ref: 
        https://openmm.github.io/openmm-cookbook/dev/notebooks/restraints_constraints_forces/Restraining%20Atom%20Positions.html '''
    
    atom_name=kwargs['atom_name']
    force_constants=kwargs['force_constants']
    restraint_x=kwargs['restraint_x']
    restraint_y=kwargs['restraint_y']
    restraint_z=kwargs['restraint_z']
    
    if (restraint_x, restraint_y, restraint_z) == (True, True, True):
        restraint = CustomExternalForce('k*periodicdistance(x, y, z, x0, y0, z0)^2')
        
        restraint.addGlobalParameter('k', force_constants*kilojoules_per_mole/nanometer)
        restraint.addPerParticleParameter('x0')
        restraint.addPerParticleParameter('y0')
        restraint.addPerParticleParameter('z0')
        
        restraint.setForceGroup(3)  #ATTENTION
        system.addForce(restraint)

        for atom in top.topology.atoms():
            if atom.name in atom_name:
                print(atom.name)
                restraint.addParticle(atom.index, gro.positions[atom.index])
                
    elif (restraint_x, restraint_y, restraint_z) == (True, True, False):
        restraint_xy = CustomExternalForce('k_xy*periodicdistance(x, y, z0, x0, y0, z0)^2')
        
        restraint_xy.addGlobalParameter('k_xy', force_constants*kilojoules_per_mole/nanometer)
        restraint_xy.addPerParticleParameter('x0')
        restraint_xy.addPerParticleParameter('y0')
        restraint_xy.addPerParticleParameter('z0')
        
        restraint_xy.setForceGroup(4)  #ATTENTION
        system.addForce(restraint_xy)

        for atom in top.topology.atoms():
            if atom.name in atom_name:
                print(atom.name)
                restraint_xy.addParticle(atom.index, gro.positions[atom.index])
    
