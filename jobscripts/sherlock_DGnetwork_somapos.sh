#!/bin/bash
#
#SBATCH -J DGnetwork_somapos
#SBATCH -o ./results/DGnetwork_somapos.%j.o
#SBATCH --nodes=1
#SBATCH --mem=63240
#SBATCH -t 3:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

set -x

module load matlab

matlab -nojvm -r 'run ./jobscripts/sherlock_DGnetwork_somapos.m'
