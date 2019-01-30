#!/bin/bash
#
#SBATCH -J dentate_Test_GC_subnet
#SBATCH -o ./results/dentate_Test_GC_subnet.%j.o
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

results_path=$SCRATCH/dentate/results/Test_GC_subnet_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun -np 12 python2.7 ./scripts/main.py \
    --config-file=Full_Scale_GC_Exc_Sat_LN.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH/dentate" \
    --results-path=$results_path \
    --io-size=2 \
    --tstop=10 \
    --v-init=-75 \
    --max-walltime-hours=1.0 \
    --dry-run \
    --cell-selection-path ./datasets/GC_subnet.yaml \
    --spike-input-path=$SCRATCH/dentate/results/Full_Scale_GC_Exc_Sat_LN_9533687.bw/dentatenet_Full_Scale_GC_Exc_Sat_LN_results.h5 \
    --spike-input-namespace='Spike Events' \
    --dry-run \
    --verbose

