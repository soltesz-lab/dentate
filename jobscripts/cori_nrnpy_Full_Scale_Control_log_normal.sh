#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_Control_log_normal
#SBATCH -o ./results/dentate_Full_Scale_Control_log_normal.%j.o
#SBATCH -N 64
#SBATCH --ntasks-per-node=32
#SBATCH -p regular
#SBATCH -t 6:30:00
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

srun -n 2048 python main.py \
 --config-file=config/Full_Scale_Control_log_normal_weights.yaml \
 --template-paths=../dgc/Mateos-Aparicio2014 \
 --dataset-prefix="$SCRATCH/dentate" \
 --results-path=$results_path \
 --io-size=256 \
 --tstop=1500 \
 --v-init=-65 \
 --max-walltime-hours=5.4 \
 --results-write-time=250 \
 --verbose
