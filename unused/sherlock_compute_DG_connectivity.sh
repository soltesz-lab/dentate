#!/bin/bash
#
#SBATCH -J compute_DG_connectivity
#SBATCH -o ./results/compute_DG_connectivity.%j.o
#SBATCH -n 8
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

set -x

mpirun -np 8 python ./scripts/compute_DG_connectivity.py

