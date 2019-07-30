#!/bin/bash

#SBATCH -J dentate_Full_Scale_GC_Exc_Sat_DD_LNN_Diag # Job name
#SBATCH -o ./results/dentate.o%j       # Name of stdout output file
#SBATCH -e ./results/dentate.e%j       # Name of stderr error file
#SBATCH -p skx-normal      # Queue (partition) name
#SBATCH -N 128             # Total # of nodes 
#SBATCH -n 6144            # Total # of mpi tasks
#SBATCH -t 12:00:00        # Run time (hh:mm:ss)
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=all    # Send email at begin and end of job

module unload python2
module load python3
module load phdf5/1.8.16

set -x

export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH

results_path=$WORK/dentate/results/Full_Scale_GC_Exc_Sat_DD_LNN_Diag_$SLURM_JOB_ID
export results_path

cd $SLURM_SUBMIT_DIR

mkdir -p $results_path

git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun python3.7 ./scripts/main.py  \
    --config-file=Full_Scale_GC_Exc_Sat_DD_LNN_Diag.yaml  \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$WORK/dentate" \
    --results-path=$results_path \
    --io-size=256 \
    --tstop=5000 \
    --v-init=-75 \
    --results-write-time=600 \
    --stimulus-onset=50.0 \
    --max-walltime-hours=11.9 \
    --vrecord-fraction=0.001 \
    --node-rank-file=parts.6144 \
    --verbose

