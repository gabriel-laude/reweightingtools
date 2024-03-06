# reweightingtools
Software package collection for dynamical Girsanov path reweighting  

## Software
- simulation programs: [openmmtools](https://github.com/bkellerlab/openmmtools)
  - reporter function: [ReweightingReporter](https://github.com/bkellerlab/reweightingtools/simulation/reweightingreporter.py)
- analysis package: [deeptime](https://github.com/bkellerlab/deeptime)

### Setup
1. download existing software with reweighting modification

   a) go to the respective [GitHub repository](https://github.com/bkellerlab/deeptime) 

   b) clone the respective package by
   	- copying the "HTTPS clone URL" link that appears after selecting the code icon and 
   	- run `git clone [url]` in the directory you want to download it to
   	  
3. create new environment (e.g. RWGHTSoftware) with `conda create -n RWGHTSoftware python=3.11` 
4. install the package and all corresponding dependencies, e.g. `conda install -c anaconda matplotlib`, `conda install -c anaconda spyder`

   a) openmmtools: install the openmmtools extension in cloned openmmtools folder via `pip insatll .`, further possible dependencies are
   
     `conda install -c conda-forge openmm`

     `conda install -c conda-forge mdtraj`

     `conda install -c conda-forge netcdf4`

     `conda install -c conda-forge mpiplus`

     `conda install -c omnia pymbar`

     `conda install -c numba numba`
   
   b) deeptime: install the deeptime extension in cloned deeptime folder via `pip insatll .`  
   
5. create scripts for simulation and analysis with Girsanov reweighting or use the setup and templates in [reweightingtools](https://github.com/bkellerlab/reweightingtools)

## Girsanov path reweighting
Dynamical reweighting methods are becoming more and more important to study complex mechanisms of chemical processes, whose direct simulations of the full state-space at the correct statistical weight are impossible. 
In order to gain a better understanding of the applicability of Girsanov path reweighting to chemical processes in complex environments, a broad network between already high-quality methods must be created.
The [article](???) presents the basis for how Girsanov path reweighting can be integrated into any simulation and analysis software. 
We also provide a ready-to-use simulation program that enables on-the-fly estimation of reweighting factors and a unified procedure for estimating a reweighted dynamical model from this data.
