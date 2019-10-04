#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_GC_Exc_Sat_LNN
#SBATCH -o ./results/dentate_Full_Scale_GC_Exc_Sat_LNN.%j.o
#SBATCH -N 320
#SBATCH --ntasks-per-node=32
#SBATCH --qos=regular
#SBATCH -t 6:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module swap PrgEnv-intel/6.0.4 PrgEnv-intel/6.0.5
module unload darshan
module load cray-hdf5-parallel
module load python/2.7-anaconda-5.2

results_path=./results/Full_Scale_GC_Exc_Sat_LNN_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export PYTHONPATH=$HOME/model:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnintel/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/.local/cori/2.7-anaconda-5.2/lib/python2.7/site-packages:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.6
export HDF5_USE_FILE_LOCKING=FALSE

echo python is `which python`

set -x

srun -n 10240 -c 2 python ./scripts/main.py \
 --config-file=Full_Scale_GC_Exc_Sat_LNN.yaml \
 --template-paths=../dgc/Mateos-Aparicio2014:templates \
 --dataset-prefix="$SCRATCH/dentate" \
 --results-path=$results_path \
 --io-size=256 \
 --tstop=1200 \
 --v-init=-75 \
 --vrecord-fraction=0.001 \
 --max-walltime-hours=5.4 \
 --results-write-time=600 \
 --ldbal \
 --lptbal \
 --verbose
