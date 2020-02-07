#!/bin/bash
#
#SBATCH -J distribute_GC_synapses
#SBATCH -o ./results/distribute_GC_synapses.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`
echo python is `which python3`

#Run the job using mpirun_rsh
mpirun_rsh -export-all -hostfile $SLURM_NODEFILE -np 1536 \
`which python3` ./scripts/distribute_synapse_locs.py \
    --distribution=poisson \
    --config-prefix=./config \
    --config=Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates --populations=GC \
    --forest-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_normalized_20200203_compressed.h5 \
    --output-path=$SCRATCH/dentate/Full_Scale_Control/DGC_forest_syns_20200203.h5 \
    --io-size=128 --write-size=10 --cache-size=$((2 * 1024 * 1024)) \
    --chunk-size=50000 --value-chunk-size=100000 -v
