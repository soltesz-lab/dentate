#!/bin/bash
#
#SBATCH -J dentate_Test_GC_1000
#SBATCH -o ./results/dentate_Test_GC_1000.%j.o
#SBATCH --nodes=12
#SBATCH --ntasks-per-node=12
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5

set -x

export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/Test_GC_1000_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin


ibrun -np 48 python3 ./scripts/main.py \
 --config-file=Test_GC_1000.yaml  \
 --arena-id=A --trajectory-id=Diag \
 --template-paths=../dgc/Mateos-Aparicio2014:templates \
 --dataset-prefix="/oasis/scratch/comet/iraikov/temp_project/dentate" \
 --results-path=$results_path \
 --io-size=24 \
 --tstop=6 \
 --v-init=-75 \
 --max-walltime-hours=1 \
 --node-rank-file=parts.12 \
 --verbose
