#!/bin/bash

#SBATCH -J repack_syns_GC        # Job name
#SBATCH -o ./results/repack_syns_GC.o%j       # Name of stdout output file
#SBATCH -e ./results/repack_syns_GC.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH -n 56            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load python3
module load phdf5

set -x

export prefix=$SCRATCH/striped/dentate/Full_Scale_Control
export input=$prefix/DGC_forest_syns_20201217.h5
export output=$prefix/DGC_forest_syns_20201217_compressed.h5

export H5TOOLS_BUFSIZE=$(( 64 * 1024 * 1024 * 1024))

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output
