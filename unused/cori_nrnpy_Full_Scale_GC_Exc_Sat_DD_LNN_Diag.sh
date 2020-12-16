#!/bin/bash
#
#SBATCH -J dentate_Full_Scale_GC_Exc_Sat_DD_LNN_Diag
#SBATCH -o ./results/dentate_Full_Scale_GC_Exc_Sat_DD_LNN_Diag.%j.o
#SBATCH -N 320
#SBATCH --ntasks-per-node=32
#SBATCH --qos=regular
#SBATCH -t 6:30:00
#SBATCH -L SCRATCH   # Job requires $SCRATCH file system
#SBATCH -C haswell   # Use Haswell nodes
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module unload craype-hugepages2M
#module swap PrgEnv-intel PrgEnv-gnu
module load cray-hdf5-parallel/1.10.5.0
module load python/3.7-anaconda-2019.07

results_path=$SCRATCH/dentate/results/Full_Scale_GC_Exc_Sat_DD_LNN_Diag_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

export PYTHONPATH=$HOME/model:$PYTHONPATH
export PYTHONPATH=$HOME/bin/nrnintel3/lib/python:$PYTHONPATH
export LD_PRELOAD=/lib64/libreadline.so.7
export HDF5_USE_FILE_LOCKING=FALSE

echo python is `which python`

set -x

srun -n 10240 -c 2 python ./scripts/main.py \
     --config-file=Full_Scale_GC_Exc_Sat_DD_LNN.yaml \
      --arena-id=A --trajectory-id=Diag \
      --template-paths=../dgc/Mateos-Aparicio2014:templates \
      --dataset-prefix="$SCRATCH/dentate" \
      --results-path=$results_path \
      --io-size=256 \
      --tstop=1200 \
      --v-init=-75 \
      --vrecord-fraction=0.001 \
      --max-walltime-hours=5.4 \
      --results-write-time=600 \
      --verbose
