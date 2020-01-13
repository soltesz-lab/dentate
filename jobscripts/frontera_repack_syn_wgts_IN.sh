#!/bin/bash

#SBATCH -J repack_syn_wgts        # Job name
#SBATCH -o ./results/repack_syn_wgts_IN.o%j       # Name of stdout output file
#SBATCH -e ./results/repack_syn_wgts_IN.e%j       # Name of stderr error file
#SBATCH -p development      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH -n 48            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job


module load phdf5

set -x

export prefix=$SCRATCH/striped/dentate/Full_Scale_Control
export input=$prefix/DG_IN_syn_weights_SLN_20200112.h5
export output=$prefix/DG_IN_syn_weights_SLN_20200112_compressed.h5

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output
