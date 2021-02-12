#!/bin/bash

#SBATCH -J RunNetclampGom # Job name
#SBATCH -o ./results/netclamp/netclampgo.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp/netclampgo.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 1            # Total # of nodes 
#SBATCH --ntasks-per-node=1 # # of mpi tasks per node
#SBATCH -t 3:30:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job

python3 RunNetclampGo.py 
