#!/bin/bash
#
#SBATCH -J dentate_cut_slice
#SBATCH -o ./results/dentate_cut_slice.%j.o
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#
. $HOME/comet_env.sh

ulimit -c unlimited

export SLURM_NODEFILE=`generate_pbs_nodefile`

set -x

results_path=$SCRATCH/dentate/results/DG_slice_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 48 \
    python3 ./scripts/cut_slice.py \
    --arena-id=A --trajectory-id=Diag \
    --config=Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --config-prefix=./config \
    --dataset-prefix="$SCRATCH/dentate" \
    --output-path=$results_path \
    --spike-input-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20190912_compressed.h5" \
    --spike-input-namespace='Input Spikes A Diag' \
    --distance-limits -25 25 \
    --write-selection \
    --verbose

