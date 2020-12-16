#!/bin/bash
#
#SBATCH -J generate_MC_extent_input_spike_trains
#SBATCH -o ./results/generate_MC_extent_input_spike_trains.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


. $HOME/comet_env.sh


set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

#Run the job using mpirun_rsh
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 10 \
    python3 $HOME/model/dentate/scripts/generate_input_spike_trains.py -p MC \
    --config=Full_Scale_Basis.yaml \
    --config-prefix=./config \
    --selectivity-path=${SCRATCH}/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201001_compressed.h5 \
    --output-path=${SCRATCH}/dentate/Slice/MC_extent_input_spike_trains_20201001.h5 \
    --arena-id A --n-trials 5 \
     -v


