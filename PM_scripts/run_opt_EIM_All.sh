#!/bin/bash

#SBATCH -J optimize_DG_network_neg2000_neg1500um # Job name
#SBATCH -o ./results/netclamp/opt_EIM_all.o%j       # Name of stdout output file
#SBATCH -e ./results/netclamp/opt_EIM_all.e%j       # Name of stderr error file
#SBATCH -p normal      # Queue (partition) name
#SBATCH -N 1            # Total # of nodes 
#SBATCH --ntasks-per-node=56 # # of mpi tasks per node
#SBATCH -t 2:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=pmoolcha@stanford.edu
#SBATCH --mail-type=all    # Send email at begin and end of job


sh PM_jobscripts/PM_netclamp_opt_EIM_000_all.sh
sh PM_jobscripts/PM_netclamp_opt_EIM_100_all.sh
#sh PM_jobscripts/PM_netclamp_opt_EIM_050_all.sh
sh PM_jobscripts/PM_netclamp_opt_EIM_025_all.sh
sh PM_jobscripts/PM_netclamp_opt_EIM_075_all.sh
