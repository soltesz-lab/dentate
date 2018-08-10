#!/bin/bash
#
#SBATCH -J dentate_Test_GC_1000
#SBATCH -o ./results/dentate_Test_GC_1000.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -p shared
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load python
module load hdf5
module load scipy
module load mpi4py

set -x

export PYTHONPATH=/share/apps/compute/mpi4py/mvapich2_ib/lib/python2.7/site-packages:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnpython/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
export LD_PRELOAD=$MPIHOME/lib/libmpi.so

results_path=$SCRATCH/dentate/results/Test_GC_1000_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin


ibrun -np 12 python ./scripts/main.py \
 --config-file=Test_GC_1000.yaml  \
 --template-paths=../dgc/Mateos-Aparicio2014 \
 --dataset-prefix="/oasis/scratch/comet/iraikov/temp_project/dentate" \
 --results-path=$results_path \
 --io-size=4 \
 --tstop=6 \
 --v-init=-75 \
 --max-walltime-hours=1 \
 --verbose
