#!/bin/bash
#
#SBATCH -J generate_structured_weights_DG_GC
#SBATCH -o ./results/generate_structured_weights_DG_GC.%j.o
#SBATCH --nodes=64
#SBATCH --ntasks-per-node=24
#SBATCH -t 0:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


. $HOME/comet_env.sh
module load hdf5 python

ulimit -c unlimited

set -x

#export SLURM_NODEFILE=`generate_pbs_nodefile`
echo python is `which python3`

export DATA_PREFIX=$SCRATCH/dentate/Full_Scale_Control

#Run the job using mpirun_rsh

ibrun `which python3` $HOME/model/dentate/scripts/generate_structured_weights_as_cell_attr.py \
    -d GC -s MPP -s LPP -n MC \
    --config=./config/Full_Scale_GC_Exc_Sat_DD_SLN.yaml \
    --initial-weights-namespace='Log-Normal Weights' \
    --initial-weights-path=$DATA_PREFIX/DG_GC_syn_weights_LN_20200708_compressed.h5 \
    --non-structured-weights-namespace='Normal Weights' \
    --non-structured-weights-path=$DATA_PREFIX/DG_GC_syn_weights_LN_20200708_compressed.h5 \
    --output-weights-namespace='Structured Weights' \
    --output-weights-path=$DATA_PREFIX/DG_GC_syn_weights_S_20200911.h5 \
    --connections-path=$DATA_PREFIX/DG_GC_connections_20200703_compressed.h5 \
    --input-features-path=$DATA_PREFIX/DG_input_features_20200910_compressed.h5 \
    --arena-id=A --optimize-tol 1e-3 --optimize-grad --arena-margin=0.3 \
    --max-delta-weight=10 --max-weight-decay-fraction=0.5 --target-amplitude=10 \
    --io-size=24 --cache-size=1  --value-chunk-size=100000 --chunk-size=20000 \
    --write-size=4 -v --dry-run --debug

# --dry-run --debug



