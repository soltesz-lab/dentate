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

export prefix=$SCRATCH/striped2/dentate/Full_Scale_Control
export input=$prefix/DG_MC_syn_weights_LN_20211201.h5
export output=$prefix/DG_MC_syn_weights_LN_20211201_compressed.h5

export H5TOOLS_BUFSIZE=$((16 * 1024 * 1024 * 1024))

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output
