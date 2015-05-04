#!/bin/bash
#
#$ -q asom,som,free64,pub64
#$ -t 1-100
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N DGC_forest_dat_export
#$ -o ./results/forest_dat_export.$JOB_ID.o
#$ -v BATCH_SIZE=100

module load MATLAB/r2014a

#INPUT_FILE=/pub/iraikov/dentate/Slice_50_300_Control/Slice_Trees.mtr
#OUTPUT_DIR=/pub/iraikov/dentate/Slice_50_300_Control

for forest in `seq 1 250`; do

SYN_INPUT_FILE=/som/iraikov/Trees_Tapered/$forest.mtr
OUTPUT_DIR=/pub/iraikov/DGC_forest/$forest

mkdir -p $OUTPUT_DIR

export INPUT_FILE
export OUTPUT_DIR

time /data/users/iraikov/model/DGC_forest/serial_dat_export

done
