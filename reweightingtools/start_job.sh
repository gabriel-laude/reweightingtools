#!/bin/bash

folder="waterbox_ABOBA_T_300"; ## name of folder that is created on scratch 

###############################################################################################################
###############################################################################################################

##--- get Current path ----------------------------------------------------
path=$(pwd);


##--- Create directory on scratch -----------------------------------------
## Check whether directory already exist: if yes: abbort, if no: continue
if [ -d "/scratch/SKieninger/$folder" ]
then
	echo "Directory does already exist."; exit 0;
else
	mkdir /scratch/SKieninger/$folder;
fi;

##--- Copy input to scratch -----------------------------------------------
cp -r input /scratch/SKieninger/$folder/.;
cp mdrun.py /scratch/SKieninger/$folder/.;


##--- Go to folder, create output folder ----------------------------------
cd /scratch/SKieninger/$folder;
mkdir output;


##--- Start Job -----------------------------------------------------------
export OPENMM_CPU_THREADS=2;
python3 mdrun.py;

##--- Copy Results to home ------------------------------------------------
cp -r output $path;

##--- Clean-up scratch ----------------------------------------------------
cd ..;
rm -r $folder;

##--- Delete all variables ------------------------------------------------
unset folder;
unset path;
