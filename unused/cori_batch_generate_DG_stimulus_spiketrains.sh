#!/bin/bash -l

#SBATCH -N 2
#SBATCH -p debug
#SBATCH -t 00:30:00
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -J generate_DG_PP_spiketrains_20171105
#SBATCH -e generate_DG_PP_spiketrains_20171105.err
#SBATCH -o generate_DG_PP_spiketrains_20171105.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=aaronmil@stanford.edu

source $HOME/.bash_profile

set -x

export trajectory=Diag

srun -n 128 python $HOME/dentate/scripts/generate_DG_stimulus_spike_trains.py \
     --config=Full_Scale_Pas.yaml \
     --features-path=$SCRATCH/Full_Scale_Control/DG_stimulus_20190610.h5 \
     --output-path=$SCRATCH/Full_Scale_Control/DG_stimulus_20190610.h5 \
     --arena-id=A \
     --trajectory-id=${trajectory} \
     -p LPP -p MPP \
     --io-size=2 -v

