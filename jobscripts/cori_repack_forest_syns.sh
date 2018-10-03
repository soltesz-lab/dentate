#!/bin/bash
#
#SBATCH -J repack_GC_forest_syns
#SBATCH -o ./results/repack_GC_forest_syns.%j.o
#SBATCH -N 1
#PBS -N repack_GC_forest_syns
#SBATCH -p regular
#SBATCH -t 12:00:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#


module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel

set -x

export prefix=$SCRATCH/dentate/Full_Scale_Control
export input=$prefix/DGC_forest_syns_20180809.h5
export output=$prefix/DGC_forest_syns_20180809_compressed.h5

srun h5repack -v -f SHUF -f GZIP=9 -i $input -o $output

