#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control
#SBATCH -o ./results/dentate_Full_Scale_Control.%j.o
#SBATCH -N 320
#SBATCH --ntasks-per-node=32
#SBATCH -t 5:30:00
#SBATCH -q regular
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module swap PrgEnv-intel PrgEnv-gnu
module unload darshan
module load cray-hdf5-parallel
module load python/2.7-anaconda-4.4

results_path=./results/Full_Scale_Control_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:/opt/python/lib/python2.7/site-packages:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrn/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/cori/2.7-anaconda-4.4/lib/python2.7/site-packages:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.6

echo python is `which python`

set -x

srun -n 20240 python ./scripts/main.py \
 --config-file=Full_Scale_Control.yaml  \
 --template-paths=../dgc/Mateos-Aparicio2014:templates \
 --dataset-prefix="$SCRATCH/dentate" \
 --results-path=$results_path \
 --io-size=196 \
 --tstop=2500 \
 --v-init=-75 \
 --stimulus-onset=50.0 \
 --max-walltime-hours=5.4 \
 --results-write-time=250 \
 --verbose
