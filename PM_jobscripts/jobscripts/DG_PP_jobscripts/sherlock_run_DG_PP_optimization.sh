#!/bin/bash
#
#SBATCH -J optimize_DG_PP_features
#SBATCH -n 40
#SBATCH -t 4:00:00
#SBATCH -o optimize_DG_PP_features.%j.o
#SBATCH --mail-user=dhh@stanford.edu
#SBATCH --mail-type=END
#SBATCH --mail-type=BEGIN
#


DENTATE_DIR=$HOME/soltesz-lab/dentate
CONFIG_DIR=$DENTATE_DIR/config
START=0
STOP=9

echo 'HERE....'

for i in `seq $START $STOP`;
do
    mpirun -n 4 python -m nested.optimize \
                --config-file=$CONFIG_DIR/optimize_DG_PP_config_place_module_$i.yaml \
                --pop-size=200         \
                --max-iter=50          \
                --path-length=3        \
                --disp                 \
                --output-dir='data'    \
                --label=$i &
done
wait

echo 'COMPLETE...'


