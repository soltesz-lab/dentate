#!/bin/bash
#
#SBATCH -J dentate_Test_Slice_10um
#SBATCH -o ./results/dentate_Test_Slice_10um.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -p compute
#SBATCH -t 1:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`


results_path=$SCRATCH/dentate/results/Test_Slice_10um_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 24 \
    python3 ./scripts/main.py \
    --config-file=Test_Slice_10um.yaml  \
    --arena-id=A --trajectory-id=Diag \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="/oasis/scratch/comet/iraikov/temp_project/dentate" \
    --results-path=$results_path \
    --io-size=24 \
    --tstop=50 \
    --v-init=-75 \
    --checkpoint-interval=10 \
    --checkpoint-clear-data \
    --max-walltime-hours=1 \
    --verbose
