#!/bin/bash

folder="pot04_temp150"; ## name of folder that is created on scratch 

###############################################################################################################
###############################################################################################################

##--- get Current path ----------------------------------------------------
path=$(pwd);


##--- Create directory on scratch -----------------------------------------
## Check whether directory already exist: if yes: abbort, if no: continue
if [ -d "/scratch/jlschaefer/$folder" ]
then
	echo "Directory does already exist."; exit 0;
else
	mkdir /scratch/jlschaefer/$folder;
fi;

##--- Copy input to scratch -----------------------------------------------
cp -r eta /scratch/jlschaefer/$folder/.;
cp run_sim.py /scratch/jlschaefer/$folder/.;
cp input_file_server.py /scratch/jlschaefer/$folder/input_file.py;
cp Integration.py /scratch/jlschaefer/$folder/.;
cp MBLP.py /scratch/jlschaefer/$folder/Potential.py;

##--- Go to folder, create output folder ----------------------------------
cd /scratch/jlschaefer/$folder;
mkdir trajs;


##--- Start Job -----------------------------------------------------------
python run_sim.py;

##--- Copy Results to home ------------------------------------------------
cp -r trajs $path;

##--- Clean-up scratch ----------------------------------------------------
cd ..;
rm -r $folder;

##--- Delete all variables ------------------------------------------------
unset folder;
unset path;
