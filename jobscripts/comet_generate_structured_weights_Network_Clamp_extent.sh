#!/bin/bash
#
#SBATCH -J generate_structured_weights_Network_Clamp
#SBATCH -o ./results/generate_structured_weights_Network_Clamp.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=24
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#

. $HOME/comet_env.sh

ulimit -c unlimited

set -x

export SLURM_NODEFILE=`generate_pbs_nodefile`

export input_file=$SCRATCH/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_extent_arena_margin_20201001.h5

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 12 \
`which python3` $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
    -d MC -s CA3c -n MC -n GC \
    --config=./config/Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    --output-weights-namespace='Structured Weights' \
    --h5types-path=$SCRATCH/dentate/Full_Scale_Control/dentate_h5types.h5 \
    --initial-weights-namespace="Log-Normal Weights" \
    --initial-weights-path=${input_file} \
    --non-structured-weights-path=${input_file} \
    --non-structured-weights-namespace="Normal Weights" \
    --output-weights-path=${input_file} \
    --output-features-path=${input_file} \
    --connections-path=${input_file} \
    --input-features-path=$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --arena-id=A  --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --coordinates 0.0 0.0 \
    --io-size=1 --cache-size=1  --value-chunk-size=100000 --chunk-size=20000 --write-size=1 \
    -v 

mpirun_rsh  -export-all -hostfile $SLURM_NODEFILE  -np 12 \
`which python3` $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -n MC \
    --config=./config/Network_Clamp_GC_Exc_Sat_SLN_extent.yaml \
    --output-weights-namespace='Structured Weights' \
    --h5types-path=$SCRATCH/dentate/Full_Scale_Control/dentate_h5types.h5 \
    --initial-weights-namespace="Log-Normal Weights" \
    --initial-weights-path=${input_file} \
    --non-structured-weights-path=${input_file} \
    --non-structured-weights-namespace="Normal Weights" \
    --output-weights-path=${input_file} \
    --output-features-path=${input_file} \
    --connections-path=${input_file} \
    --input-features-path=$SCRATCH/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5 \
    --arena-id=A  --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --coordinates 0.0 0.0 \
    --io-size=1 --cache-size=1  --value-chunk-size=100000 --chunk-size=20000 --write-size=1 \
    -v 

