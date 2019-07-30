#!/bin/bash
#
#SBATCH -J compute_vertex_metrics
#SBATCH -o ./results/compute_vertex_metrics.%j.o
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#


module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5

export SCRATCH=/oasis/scratch/comet/iraikov/temp_project

ulimit -c unlimited

set -x

input=$SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190717_compressed.h5 

ibrun -np 96 $HOME/src/neuroh5/build/neurograph_vertex_metrics --indegree --outdegree -i 24 \
    $input
