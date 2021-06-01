#!/bin/bash

#SBATCH -J repack_spike_trains        # Job name
#SBATCH -o ./results/repack_spike_trains.o%j       # Name of stdout output file
#SBATCH -e ./results/repack_spike_trains.e%j       # Name of stderr error file
#SBATCH -p small      # Queue (partition) name
#SBATCH -N 1             # Total # of nodes 
#SBATCH -n 56            # Total # of mpi tasks
#SBATCH -t 02:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module load phdf5

set -x

export prefix=$SCRATCH/striped/dentate/Full_Scale_Control
export input=$prefix/DG_input_spike_trains_phasemod_20210521.h5
export output=$prefix/DG_input_spike_trains_phasemod_20210521_compressed.h5

export H5TOOLS_BUFSIZE=$((64 * 1024 * 1024 * 1024))
echo H5TOOLS_BUFSIZE is $H5TOOLS_BUFSIZE

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output

