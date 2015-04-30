#!/bin/bash
#
#$ -q asom,som,free64,pub64
#$ -pe openmp 12
#$ -t 1-300
#$ -cwd
#$ -j y
#$ -S /bin/bash
#$ -N dentate_serial_export_connectivity
#$ -o ./results/serial_export_dentate_conn.$JOB_ID.o
#$ -v BATCH_SIZE=300
#$ -ckpt restart
#$ -l mem_free=350G

module load MATLAB/r2014b

SYN_INPUT_FILE=/som/iraikov/dentate/Full_Scale_Control/Syn_Connections.mat
GJ_INPUT_FILE=/som/iraikov/dentate/Full_Scale_Control/GJ_Connections.mat
OUTPUT_DIR=/som/iraikov/dentate/Full_Scale_Control

export SYN_INPUT_FILE
export GJ_INPUT_FILE
export OUTPUT_DIR

time /data/users/iraikov/model/dentate/scripts/export_connectivity_mpi

