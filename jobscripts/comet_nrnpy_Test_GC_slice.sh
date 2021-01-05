#!/bin/bash
#
#SBATCH -J dentate_Test_GC_slice
#SBATCH -o ./results/dentate_Test_GC_slice.%j.o
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 2:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

set -x

export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/Test_GC_slice_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun python3 ./scripts/main.py \
    --arena-id=A --trajectory-id=Diag \
    --config-file=Test_Slice_neg25_pos25um_IN_Izh.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH/dentate" \
    --results-path=$results_path \
    --io-size=12 \
    --tstop=9500 \
    --v-init=-75 \
    --max-walltime-hours=1 \
    --spike-input-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
    --spike-input-namespace='Input Spikes' \
    --microcircuit-inputs \
    --checkpoint-interval 0. \
    --recording-fraction 0.01 \
    --use-coreneuron \
    --verbose

