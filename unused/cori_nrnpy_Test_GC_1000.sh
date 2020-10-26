#!/bin/bash
#
#SBATCH -J dentate_Test_GC_1000
#SBATCH -o ./results/dentate_Test_GC_1000.%j.o
#SBATCH -N 8
#SBATCH --ntasks-per-node=24
#SBATCH -q debug
#SBATCH -t 0:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module unload craype-hugepages2M
#module swap PrgEnv-intel PrgEnv-gnu
module load cray-hdf5-parallel/1.10.5.0
module load python/3.7-anaconda-2019.07

results_path=$SCRATCH/dentate/results/Test_GC_1000_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export PYTHONPATH=$HOME/model:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnintel3/lib/python:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.7
export HDF5_USE_FILE_LOCKING=FALSE

echo python is `which python`
echo PYTHONPATH is $PYTHONPATH

set -x

srun -n 192 -c 2 python ./scripts/main.py \
 --config-file=Test_GC_1000.yaml \
 --arena-id=A --trajectory-id=Diag \
 --template-paths=../dgc/Mateos-Aparicio2014:templates \
 --dataset-prefix="$SCRATCH/dentate" \
 --results-path=$results_path \
 --io-size=48 \
 --tstop=150 \
 --v-init=-75 \
 --vrecord-fraction=0.001 \
 --max-walltime-hours=0.48 \
 --results-write-time=600 \
 --verbose
