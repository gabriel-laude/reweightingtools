''' compare 100 step trajectory of different implementation and different bias types
Mueller Brown potential with linear and polynomial bias
'''
from input_CI_LSG import *
#%% openMM custom implementation   
reference_directory = './openMM_LSG/ICaI090_5z/'
print(reference_directory)
output_directory = './openMM_CI/ICaI090_5z/'

temperature=T*kelvin
collision_rate=xi*(picoseconds)**(-1)
timestep=step*femtoseconds
nonbondedCutoff=nbC*nanometer

# load starting position, velocity, eta from reference calculation
startingPositions=np.load(reference_directory+'startingPositions.npy')
#print(startingPositions.shape)
#startingPositions=startingPositions.reshape(nsteps,sol+mol,3)
#startingPositions=startingPositions[0]
startingVelocities=np.load(reference_directory+'startingVelocities.npy')
#startingVelocities=startingVelocities.reshape(nsteps,sol+mol,3)
#startingVelocities=startingVelocities[0]
referenceEta=np.loadtxt(reference_directory+'eta.txt')
referenceEta=referenceEta.reshape(nsteps,sol+mol,3)

# set ff
gro = GromacsGroFile(inputMD + str(gro_input) + '.gro')
top = GromacsTopFile(inputMD + str(top_input) + '.top',
                     periodicBoxVectors=gro.getPeriodicBoxVectors(),
                     includeDir=forcefield+'/'
                        )
# Defining Simulation Dependencies 
platform = Platform.getPlatformByName(PlatformByNamen)    
plumedScript = open(plumedMD+plumed_file, 'r')
plumedScript = plumedScript.read()
plumedScript = plumedScript%(output_directory+colvar_file,nstxout)
externalForce=plumedMD+externalForce_file

# set eta from reference calculation
eta = referenceEta
eta100 = eta[100]
eta99 = eta[99]
eta98 = eta[98] 
eta97 = eta[97] 
eta96 = eta[96] 
eta95 = eta[95] 
eta94 = eta[94] 
eta93 = eta[93] 
eta92 = eta[92] 
eta91 = eta[91] 
eta90 = eta[90] 
eta89 = eta[89]
eta88 = eta[88] 
eta87 = eta[87] 
eta86 = eta[86] 
eta85 = eta[85] 
eta84 = eta[84] 
eta83 = eta[83] 
eta82 = eta[82] 
eta81 = eta[81] 
eta80 = eta[80] 
eta79 = eta[79] 
eta78 = eta[78] 
eta77 = eta[77] 
eta76 = eta[76] 
eta75 = eta[75] 
eta74 = eta[74] 
eta73 = eta[73] 
eta72 = eta[72] 
eta71 = eta[71] 
eta70 = eta[70] 
eta69 = eta[69] 
eta68 = eta[68] 
eta67 = eta[67] 
eta66 = eta[66] 
eta65 = eta[65] 
eta64 = eta[64] 
eta63 = eta[63] 
eta62 = eta[62] 
eta61 = eta[61] 
eta60 = eta[60] 
eta59 = eta[59] 
eta58 = eta[58] 
eta57 = eta[57] 
eta56 = eta[56] 
eta55 = eta[55] 
eta54 = eta[54] 
eta53 = eta[53] 
eta52 = eta[52] 
eta51 = eta[51] 
eta50 = eta[50] 
eta49 = eta[49] 
eta48 = eta[48] 
eta47 = eta[47] 
eta46 = eta[46] 
eta45 = eta[45] 
eta44 = eta[44] 
eta43 = eta[43] 
eta42 = eta[42] 
eta41 = eta[41] 
eta40 = eta[40] 
eta39 = eta[39] 
eta38 = eta[38] 
eta37 = eta[37] 
eta36 = eta[36] 
eta35 = eta[35] 
eta34 = eta[34] 
eta33 = eta[33] 
eta32 = eta[32] 
eta31 = eta[31] 
eta30 = eta[30] 
eta29 = eta[29] 
eta28 = eta[28] 
eta27 = eta[27] 
eta26 = eta[26] 
eta25 = eta[25] 
eta24 = eta[24] 
eta23 = eta[23] 
eta22 = eta[22] 
eta21 = eta[21] 
eta20 = eta[20] 
eta19 = eta[19] 
eta18 = eta[18] 
eta17 = eta[17] 
eta16 = eta[16] 
eta15 = eta[15] 
eta14 = eta[14] 
eta13 = eta[13] 
eta12 = eta[12] 
eta11 = eta[11] 
eta10 = eta[10]
eta9 = eta[9]
eta8 = eta[8]
eta7 = eta[7]
eta6 = eta[6]
eta5 = eta[5]
eta4 = eta[4]
eta3 = eta[3]
eta2 = eta[2]
eta1 = eta[1]
eta0 = eta[0]

# logging    
log = open(output_directory + "/MD.log", 'a')
log.write("Simulation Start " + str(dati.now()) + "\n" )
log.write('potential        :  ' + str(forcefield) + " \n")
log.write('perturbation     :  ' + str(plumedMD+externalForce_file) + " \n")
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


Integrator.addPerDofVariable("Eta1",0)
Integrator.addPerDofVariable("Eta2",0)
Integrator.addPerDofVariable("Eta3",0)
Integrator.addPerDofVariable("Eta4",0)
Integrator.addPerDofVariable("Eta5",0)
Integrator.addPerDofVariable("Eta6",0)
Integrator.addPerDofVariable("Eta7",0)
Integrator.addPerDofVariable("Eta8",0)
Integrator.addPerDofVariable("Eta9",0)
Integrator.addPerDofVariable("Eta10",0)
Integrator.addPerDofVariable("Eta11",0)
Integrator.addPerDofVariable("Eta12",0)
Integrator.addPerDofVariable("Eta13",0)
Integrator.addPerDofVariable("Eta14",0)
Integrator.addPerDofVariable("Eta15",0)
Integrator.addPerDofVariable("Eta16",0)
Integrator.addPerDofVariable("Eta17",0)
Integrator.addPerDofVariable("Eta18",0)
Integrator.addPerDofVariable("Eta19",0)
Integrator.addPerDofVariable("Eta20",0)
Integrator.addPerDofVariable("Eta21",0)
Integrator.addPerDofVariable("Eta22",0)
Integrator.addPerDofVariable("Eta23",0)
Integrator.addPerDofVariable("Eta24",0)
Integrator.addPerDofVariable("Eta25",0)
Integrator.addPerDofVariable("Eta26",0)
Integrator.addPerDofVariable("Eta27",0)
Integrator.addPerDofVariable("Eta28",0)
Integrator.addPerDofVariable("Eta29",0)
Integrator.addPerDofVariable("Eta30",0)
Integrator.addPerDofVariable("Eta31",0)
Integrator.addPerDofVariable("Eta32",0)
Integrator.addPerDofVariable("Eta33",0)
Integrator.addPerDofVariable("Eta34",0)
Integrator.addPerDofVariable("Eta35",0)
Integrator.addPerDofVariable("Eta36",0)
Integrator.addPerDofVariable("Eta37",0)
Integrator.addPerDofVariable("Eta38",0)
Integrator.addPerDofVariable("Eta39",0)
Integrator.addPerDofVariable("Eta40",0)
Integrator.addPerDofVariable("Eta41",0)
Integrator.addPerDofVariable("Eta42",0)
Integrator.addPerDofVariable("Eta43",0)
Integrator.addPerDofVariable("Eta44",0)
Integrator.addPerDofVariable("Eta45",0)
Integrator.addPerDofVariable("Eta46",0)
Integrator.addPerDofVariable("Eta47",0)
Integrator.addPerDofVariable("Eta48",0)
Integrator.addPerDofVariable("Eta49",0)
Integrator.addPerDofVariable("Eta50",0)
Integrator.addPerDofVariable("Eta51",0)
Integrator.addPerDofVariable("Eta52",0)
Integrator.addPerDofVariable("Eta53",0)
Integrator.addPerDofVariable("Eta54",0)
Integrator.addPerDofVariable("Eta55",0)
Integrator.addPerDofVariable("Eta56",0)
Integrator.addPerDofVariable("Eta57",0)
Integrator.addPerDofVariable("Eta58",0)
Integrator.addPerDofVariable("Eta59",0)
Integrator.addPerDofVariable("Eta60",0)
Integrator.addPerDofVariable("Eta61",0)
Integrator.addPerDofVariable("Eta62",0)
Integrator.addPerDofVariable("Eta63",0)
Integrator.addPerDofVariable("Eta64",0)
Integrator.addPerDofVariable("Eta65",0)
Integrator.addPerDofVariable("Eta66",0)
Integrator.addPerDofVariable("Eta67",0)
Integrator.addPerDofVariable("Eta68",0)
Integrator.addPerDofVariable("Eta69",0)
Integrator.addPerDofVariable("Eta70",0)
Integrator.addPerDofVariable("Eta71",0)
Integrator.addPerDofVariable("Eta72",0)
Integrator.addPerDofVariable("Eta73",0)
Integrator.addPerDofVariable("Eta74",0)
Integrator.addPerDofVariable("Eta75",0)
Integrator.addPerDofVariable("Eta76",0)
Integrator.addPerDofVariable("Eta77",0)
Integrator.addPerDofVariable("Eta78",0)
Integrator.addPerDofVariable("Eta79",0)
Integrator.addPerDofVariable("Eta80",0)
Integrator.addPerDofVariable("Eta81",0)
Integrator.addPerDofVariable("Eta82",0)
Integrator.addPerDofVariable("Eta83",0)
Integrator.addPerDofVariable("Eta84",0)
Integrator.addPerDofVariable("Eta85",0)
Integrator.addPerDofVariable("Eta86",0)
Integrator.addPerDofVariable("Eta87",0)
Integrator.addPerDofVariable("Eta88",0)
Integrator.addPerDofVariable("Eta89",0)
Integrator.addPerDofVariable("Eta90",0)
Integrator.addPerDofVariable("Eta91",0)
Integrator.addPerDofVariable("Eta92",0)
Integrator.addPerDofVariable("Eta93",0)
Integrator.addPerDofVariable("Eta94",0)
Integrator.addPerDofVariable("Eta95",0)
Integrator.addPerDofVariable("Eta96",0)
Integrator.addPerDofVariable("Eta97",0)
Integrator.addPerDofVariable("Eta98",0)
Integrator.addPerDofVariable("Eta99",0)
Integrator.addPerDofVariable("Eta100",0)
    
Integrator.setPerDofVariableByName("Eta0",eta0)
Integrator.setPerDofVariableByName("Eta1",eta1)
Integrator.setPerDofVariableByName("Eta2",eta2)
Integrator.setPerDofVariableByName("Eta3",eta3)
Integrator.setPerDofVariableByName("Eta4",eta4)
Integrator.setPerDofVariableByName("Eta5",eta5)
Integrator.setPerDofVariableByName("Eta6",eta6)
Integrator.setPerDofVariableByName("Eta7",eta7)
Integrator.setPerDofVariableByName("Eta8",eta8)
Integrator.setPerDofVariableByName("Eta9",eta9)
Integrator.setPerDofVariableByName("Eta10",eta10)
Integrator.setPerDofVariableByName("Eta11",eta11)
Integrator.setPerDofVariableByName("Eta12",eta12)
Integrator.setPerDofVariableByName("Eta13",eta13)
Integrator.setPerDofVariableByName("Eta14",eta14)
Integrator.setPerDofVariableByName("Eta15",eta15)
Integrator.setPerDofVariableByName("Eta16",eta16)
Integrator.setPerDofVariableByName("Eta17",eta17)
Integrator.setPerDofVariableByName("Eta18",eta18)
Integrator.setPerDofVariableByName("Eta19",eta19)
Integrator.setPerDofVariableByName("Eta20",eta20)
Integrator.setPerDofVariableByName("Eta21",eta21)
Integrator.setPerDofVariableByName("Eta22",eta22)
Integrator.setPerDofVariableByName("Eta23",eta23)
Integrator.setPerDofVariableByName("Eta24",eta24)
Integrator.setPerDofVariableByName("Eta25",eta25)
Integrator.setPerDofVariableByName("Eta26",eta26)
Integrator.setPerDofVariableByName("Eta27",eta27)
Integrator.setPerDofVariableByName("Eta28",eta28)
Integrator.setPerDofVariableByName("Eta29",eta29)
Integrator.setPerDofVariableByName("Eta30",eta30)
Integrator.setPerDofVariableByName("Eta31",eta31)
Integrator.setPerDofVariableByName("Eta32",eta32)
Integrator.setPerDofVariableByName("Eta33",eta33)
Integrator.setPerDofVariableByName("Eta34",eta34)
Integrator.setPerDofVariableByName("Eta35",eta35)
Integrator.setPerDofVariableByName("Eta36",eta36)
Integrator.setPerDofVariableByName("Eta37",eta37)
Integrator.setPerDofVariableByName("Eta38",eta38)
Integrator.setPerDofVariableByName("Eta39",eta39)
Integrator.setPerDofVariableByName("Eta40",eta40)
Integrator.setPerDofVariableByName("Eta41",eta41)
Integrator.setPerDofVariableByName("Eta42",eta42)
Integrator.setPerDofVariableByName("Eta43",eta43)
Integrator.setPerDofVariableByName("Eta44",eta44)
Integrator.setPerDofVariableByName("Eta45",eta45)
Integrator.setPerDofVariableByName("Eta46",eta46)
Integrator.setPerDofVariableByName("Eta47",eta47)
Integrator.setPerDofVariableByName("Eta48",eta48)
Integrator.setPerDofVariableByName("Eta49",eta49)
Integrator.setPerDofVariableByName("Eta50",eta50)
Integrator.setPerDofVariableByName("Eta51",eta51)
Integrator.setPerDofVariableByName("Eta52",eta52)
Integrator.setPerDofVariableByName("Eta53",eta53)
Integrator.setPerDofVariableByName("Eta54",eta54)
Integrator.setPerDofVariableByName("Eta55",eta55)
Integrator.setPerDofVariableByName("Eta56",eta56)
Integrator.setPerDofVariableByName("Eta57",eta57)
Integrator.setPerDofVariableByName("Eta58",eta58)
Integrator.setPerDofVariableByName("Eta59",eta59)
Integrator.setPerDofVariableByName("Eta60",eta60)
Integrator.setPerDofVariableByName("Eta61",eta61)
Integrator.setPerDofVariableByName("Eta62",eta62)
Integrator.setPerDofVariableByName("Eta63",eta63)
Integrator.setPerDofVariableByName("Eta64",eta64)
Integrator.setPerDofVariableByName("Eta65",eta65)
Integrator.setPerDofVariableByName("Eta66",eta66)
Integrator.setPerDofVariableByName("Eta67",eta67)
Integrator.setPerDofVariableByName("Eta68",eta68)
Integrator.setPerDofVariableByName("Eta69",eta69)
Integrator.setPerDofVariableByName("Eta70",eta70)
Integrator.setPerDofVariableByName("Eta71",eta71)
Integrator.setPerDofVariableByName("Eta72",eta72)
Integrator.setPerDofVariableByName("Eta73",eta73)
Integrator.setPerDofVariableByName("Eta74",eta74)
Integrator.setPerDofVariableByName("Eta75",eta75)
Integrator.setPerDofVariableByName("Eta76",eta76)
Integrator.setPerDofVariableByName("Eta77",eta77)
Integrator.setPerDofVariableByName("Eta78",eta78)
Integrator.setPerDofVariableByName("Eta79",eta79)
Integrator.setPerDofVariableByName("Eta80",eta80)
Integrator.setPerDofVariableByName("Eta81",eta81)
Integrator.setPerDofVariableByName("Eta82",eta82)
Integrator.setPerDofVariableByName("Eta83",eta83)
Integrator.setPerDofVariableByName("Eta84",eta84)
Integrator.setPerDofVariableByName("Eta85",eta85)
Integrator.setPerDofVariableByName("Eta86",eta86)
Integrator.setPerDofVariableByName("Eta87",eta87)
Integrator.setPerDofVariableByName("Eta88",eta88)
Integrator.setPerDofVariableByName("Eta89",eta89)
Integrator.setPerDofVariableByName("Eta90",eta90)
Integrator.setPerDofVariableByName("Eta91",eta91)
Integrator.setPerDofVariableByName("Eta92",eta92)
Integrator.setPerDofVariableByName("Eta93",eta93)
Integrator.setPerDofVariableByName("Eta94",eta94)
Integrator.setPerDofVariableByName("Eta95",eta95)
Integrator.setPerDofVariableByName("Eta96",eta96)
Integrator.setPerDofVariableByName("Eta97",eta97)
Integrator.setPerDofVariableByName("Eta98",eta98)
Integrator.setPerDofVariableByName("Eta99",eta99)
Integrator.setPerDofVariableByName("Eta100",eta100)

Integrator.addUpdateContextState()
Integrator.addGlobalVariable('kT', kB * temperature) 
addComputeTemperatureDependentConstants(Integrator, {"sigma": "sqrt(kT/m)"}) 

# "R V O V R" : (1, ['1/(b*sigma*m) * (1 + a) * timestep/2 * ff0'], "R U V O V R")
# Add random number
Integrator.addComputePerDof("Eta0", "Eta0 * delta(n-0) + Eta1 * delta(n-1) + Eta2 * delta(n-2) + Eta3 * delta(n-3) + Eta4 * delta(n-4) + Eta5 * delta(n-5) + Eta6 * delta(n-6) + Eta7 * delta(n-7) + Eta8 * delta(n-8) + Eta9 * delta(n-9) + Eta10 * delta(n-10) + Eta11 * delta(n-11) + Eta12 * delta(n-12) + Eta13 * delta(n-13) + Eta14 * delta(n-14) + Eta15 * delta(n-15) + Eta16 * delta(n-16) + Eta17 * delta(n-17) + Eta18 * delta(n-18) + Eta19 * delta(n-19) + Eta20 * delta(n-20) + Eta21 * delta(n-21) + Eta22 * delta(n-22) + Eta23 * delta(n-23) + Eta24 * delta(n-24) + Eta25 * delta(n-25) + Eta26 * delta(n-26) + Eta27 * delta(n-27) + Eta28 * delta(n-28) + Eta29 * delta(n-29) + Eta30 * delta(n-30) + Eta31 * delta(n-31) + Eta32 * delta(n-32) + Eta33 * delta(n-33) + Eta34 * delta(n-34) + Eta35 * delta(n-35) + Eta36 * delta(n-36) + Eta37 * delta(n-37) + Eta38 * delta(n-38) + Eta39 * delta(n-39) + Eta40 * delta(n-40) + Eta41 * delta(n-41) + Eta42 * delta(n-42) + Eta43 * delta(n-43) + Eta44 * delta(n-44) + Eta45 * delta(n-45) + Eta46 * delta(n-46) + Eta47 * delta(n-47) + Eta48 * delta(n-48) + Eta49 * delta(n-49) + Eta50 * delta(n-50) + Eta51 * delta(n-51) + Eta52 * delta(n-52) + Eta53 * delta(n-53) + Eta54 * delta(n-54) + Eta55 * delta(n-55) + Eta56 * delta(n-56) + Eta57 * delta(n-57) + Eta58 * delta(n-58) + Eta59 * delta(n-59) + Eta60 * delta(n-60) + Eta61 * delta(n-61) + Eta62 * delta(n-62) + Eta63 * delta(n-63) + Eta64 * delta(n-64) + Eta65 * delta(n-65) + Eta66 * delta(n-66) + Eta67 * delta(n-67) + Eta68 * delta(n-68) + Eta69 * delta(n-69) + Eta70 * delta(n-70) + Eta71 * delta(n-71) + Eta72 * delta(n-72) + Eta73 * delta(n-73) + Eta74 * delta(n-74) + Eta75 * delta(n-75) + Eta76 * delta(n-76) + Eta77 * delta(n-77) + Eta78 * delta(n-78) + Eta79 * delta(n-79) + Eta80 * delta(n-80) + Eta81 * delta(n-81) + Eta82 * delta(n-82) + Eta83 * delta(n-83) + Eta84 * delta(n-84) + Eta85 * delta(n-85) + Eta86 * delta(n-86) + Eta87 * delta(n-87) + Eta88 * delta(n-88) + Eta89 * delta(n-89) + Eta90 * delta(n-90) + Eta91 * delta(n-91) + Eta92 * delta(n-92) + Eta93 * delta(n-93) + Eta94 * delta(n-94) + Eta95 * delta(n-95) + Eta96 * delta(n-96) + Eta97 * delta(n-97) + Eta98 * delta(n-98) + Eta99 * delta(n-99) + Eta100 * delta(n-100)")

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
system=top.createSystem(nonbondedMethod=nonbondedMethod, 
                        nonbondedCutoff=nonbondedCutoff, 
                        constraints=constraints)  # possibility to include more constraints   
nParticles = system.getNumParticles()
print(nParticles)
    
# set the bias via ExternalForce
ExternalForce = open(externalForce, 'r')
ExternalForce = ExternalForce.read()
perturbation = CustomExternalForce(ExternalForce)
perturbation.addPerParticleParameter('x0')
perturbation.addPerParticleParameter('y0')
perturbation.addPerParticleParameter('z0')   
perturbation.setForceGroup(1)  #ATTENTION
system.addForce(perturbation)
for atom in top.topology.atoms():
    if atom.name in ('CA'):  #### ATTENTION: hard coded
        print(atom.name)
        perturbation.addParticle(atom.index, gro.positions[atom.index])         
# Setting COLVAR output via PLUMED force 
perturbation2 = PlumedForce(plumedScript.rstrip())
perturbation2.setForceGroup(2)
system.addForce(perturbation2)
    
if restraints_func==None:
    pass
else:
    set_restraints(restraints_func, system, top, gro, **restraints_kwargs)
    
# set dependencies for MD simulation
MDsimulation = simulation.Simulation(top.topology, system, integrator, platform)
MDsimulation.context.setPositions(startingPositions)
MDsimulation.context.setVelocities(startingVelocities)


# write out simulation data 
output_rwgt    = "ReweightingFactors.txt"  
output_chckpt  = "chckpt"                  
MDsimulation.reporters.append(ReweightingReporter(output_directory+output_rwgt, 
                                                  nstxout, 
                                                  integrator, 
                                                  firtsPertubation=True, 
                                                  separator=' '))
MDsimulation.reporters.append(CheckpointReporter(output_directory+output_chckpt, 
                                                 nstxout))
MDsimulation.reporters.append(BiasReporter(output_directory + "bias_force.txt", nstxout, integrator, nParticles))
MDsimulation.reporters.append(PositionReporter(output_directory + "positions.txt", nstxout, nParticles))
MDsimulation.reporters.append(VelocityReporter(output_directory + "velocities.txt", nstxout, nParticles))   
MDsimulation.reporters.append(RandomNumberReporter(output_directory + "eta.txt", nstxout, integrator, nParticles))
MDsimulation.reporters.append(DeltaEtaReporter(output_directory + "DeltaEta.txt", nstxout, integrator, nParticles)) 
MDsimulation.reporters.append(TotalForceReporter(output_directory + "total_force.txt", nstxout, integrator, nParticles))
   
print('  S I M U L A T I O N  S T A R T :  %s' %(dati.now()))
MDsimulation.step(nsteps)  
print('  S I M U L A T I O N  E N D:  %s' %(dati.now()))
log = open(output_directory + "/MD.log", 'a')
log.write("Simulation End " + str(dati.now()) + "\n" )
log.close()

#%% visualise
#output_directory='./openMM_LSG/ICaI090_5z/'
#quant = np.loadtxt(output_directory+'positions.txt')
#quant=quant.reshape(101,150,3)
#quant=quant[:,0,:]
#output_directory='./openMM_CI/ICaI090_5z/'
#quant2 = np.loadtxt(output_directory+'positions.txt')
#quant2=quant2.reshape(101,150,3)
#quant2=quant2[:,0,:]
#print(quant.shape)
#print(quant2.shape)

#import reweightingtools.analysis.plotting.visualizations as vis
#x = np.arange(len(quant[:,0]))
#vis.plot_quantity([x,x],[quant[:,1],quant2[:,1]],['C1','C2'])