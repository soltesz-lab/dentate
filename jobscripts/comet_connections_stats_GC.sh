#!/bin/bash
#
#SBATCH -J generate_connections_stats_GC
#SBATCH -o ./results/connections_stats_GC.%j.o
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

set -x

ibrun -np 96 python3.5 ./scripts/connections_distance_stats.py \
       -p $SCRATCH/dentate/Full_Scale_Control/DG_GC_connections_20190430_compressed.h5  \
       -d GC -s BC -s AAC -s HC -s HCC -s MOPP -s NGFC \
       --io-size 24 -v
 
