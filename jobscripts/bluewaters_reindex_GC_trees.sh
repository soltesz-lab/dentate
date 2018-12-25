#!/bin/bash
#PBS -l nodes=40:ppn=8:xe
#PBS -q high
#PBS -l walltime=2:30:00
#PBS -e ./results/reindex_GC_trees.$PBS_JOBID.err
#PBS -o ./results/reindex_GC_trees.$PBS_JOBID.out
#PBS -N reindex_GC_trees
### set email notification
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc

echo python is `which python`

cd $PBS_O_WORKDIR

set -x

export forest_path=$SCRATCH/Full_Scale_Control/DGC_forest_extended_20181222_compressed.h5
export index_path=$SCRATCH/Full_Scale_Control/dentate_GC_coords_20180418.h5
export output_path=$SCRATCH/Full_Scale_Control/DGC_forest_reindex_20181222.h5

cp $HOME/model/dentate/datasets/dentate_h5types.h5 $output_path

aprun -n 320 -b -- bwpy-environ -- \
    python2.7 ./scripts/reindex_trees.py \
    --population=GC \
    --forest-path=$forest_path \
    --output-path=$output_path \
    --index-path=$index_path \
    --io-size=64 -v
