''' compare 100 step trajectory of different implementation and different bias types
Mueller Brown potential with linear and polynomial bias
'''
from input_CI_LSG import *
#%% LangevinSplittingGirsanov implementation
output_directory = './openMM_LSG/ICaI090_5z/'
openMM_biased_simulation(
        forcefield='amber99.ff',
        equisteps=0,
        nsteps=nsteps,
        integrator_scheme='LangevinWithGirsanov',
        integrator_splitting='R V O V R',
        temperature=T*kelvin, 
        collision_rate=xi*(picoseconds)**(-1),
        timestep=step*femtoseconds,
        nstxout=nstxout,
        outputMD=output_directory,
        PlatformByNamen=PlatformByNamen,
        gro_input=gro_input,
        top_input=top_input,
        plumed_file=plumed_file,
        externalForce_file=externalForce_file,
        colvar_file=colvar_file,
        nonbondedCutoff=nbC*nanometer,
        restraints_func=restraints_func, 
        restraints_kwargs=restraints_kwargs,
        solvent=solvent,
        fix_system=True,
        restart=False,
        ETA=True,
        dcd_output=False,
        pdp_output=False,
        data_output=False
            )
