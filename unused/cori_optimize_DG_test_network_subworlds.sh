#!/bin/bash -l

#SBATCH -J optimize_DG_test_network_subworlds_20190116
#SBATCH -o /global/cscratch1/sd/aaronmil/dentate/logs/optimize_DG_test_network_subworlds_20190116.%j.o
#SBATCH -e /global/cscratch1/sd/aaronmil/dentate/logs/optimize_DG_test_network_subworlds_20190116.%j.e
#SBATCH -q debug
#SBATCH -N 2 -n 64
#SBATCH -L SCRATCH
#SBATCH -C haswell
#SBATCH -t 00:30:00
#SBATCH --mail-user=aaronmil@stanford.edu
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $HOME/dentate/tests

srun -N 2 -n 64 -c 2 python -m nested.optimize --config-file-path=../config/cori_DG_test_network_subworlds_config.yaml --output-dir=$SCRATCH/dentate/results --pop-size=2 --max-iter=2 --path-length=1 --disp --procs-per-worker=32
