import numpy as np
import Potential as pot
from Integration import *
from PhysicalConstants import *
import matplotlib.pyplot as plt

potential           = pot.MBLP()
potential_gradient  = potential.coupled_mueller_brown_gradient

#traj_0=np.loadtxt('/home/schaefej51/Documents/2_Projects/202312_EMvsABOBA/simulation/ABOBA_300/run_5/'+'positions.txt')
#traj_1=np.loadtxt('/home/schaefej51/Documents/2_Projects/202312_EMvsABOBA/simulation/ABOBA_300/run_7/'+'positions.txt')
#traj_2=np.loadtxt('/home/schaefej51/Documents/2_Projects/202312_EMvsABOBA/simulation/ABOBA_300/run_9/'+'positions.txt')


traj_0_ds=traj_0[::100]
traj_1_ds=traj_1[::100]
traj_2_ds=traj_2[::100]


SBD_0, xedges, yedges, bin_xedges, bin_yedges = histo(traj_0_ds, 100,100)
np.save('SBD_0',SBD_0)
np.save('xedges',xedges) 
np.save('yedges',yedges)
np.save('bin_xedges',bin_xedges) 
np.save('bin_yedges',bin_yedges)
SBD_1, xedges, yedges, bin_xedges, bin_yedges = histo(traj_1_ds, bin_xedges, bin_yedges)
np.save('SBD_1',SBD_1)
SBD_2, xedges, yedges, bin_xedges, bin_yedges = histo(traj_2_ds, bin_xedges, bin_yedges)
np.save('SBD_2',SBD_2)

V_0=MB_potential(traj_0_ds[:,0],traj_0_ds[:,1])
np.save('V_0',V_0)
V_1=MB_potential(traj_1_ds[:,0],traj_1_ds[:,1])
np.save('V_1',V_1)
V_2=MB_potential(traj_2_ds[:,0],traj_2_ds[:,1])
np.save('V_2',V_2)

V=MBP_2D_disc(xedges, yedges)
np.save('V',V)

B= np.exp(-beta*V.T)
B=B/B.max()
np.save('B',B)

np.save('relBolzSamp_0',(B-SBD_0)/B)
np.save('relBolzSamp_0',(B-SBD_1)/B)
np.save('relBolzSamp_0',(B-SBD_2)/B)

