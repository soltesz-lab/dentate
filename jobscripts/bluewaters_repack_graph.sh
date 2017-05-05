#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=8:00:00
### set the job name
#PBS -N dentate_repack
### set the job stdout and stderr
#PBS -e ./results/repack_graph.$PBS_JOBID.err
#PBS -o ./results/repack_graph.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export prefix=/projects/sciteam/baef/Full_Scale_Control
export input=$prefix/dentate_Full_Scale_GC_20170501.h5
#export copy=$prefix/DGC_forest_syns_020217_copy.h5
export output=$prefix/dentate_Full_Scale_GC_20170501_compressed.h5

#aprun h5copy -v -i $input -o $copy -s /Populations -d /Populations

aprun h5repack -L -v \
-f /Projections/AACtoGC/Attributes/Edge/syn_id,\
/Projections/BCtoGC/Attributes/Edge/syn_id,\
/Projections/HCtoGC/Attributes/Edge/syn_id,\
/Projections/HCCtoGC/Attributes/Edge/syn_id,\
/Projections/MCtoGC/Attributes/Edge/syn_id,\
/Projections/MOPPtoGC/Attributes/Edge/syn_id,\
/Projections/MPPtoGC/Attributes/Edge/syn_id,\
#/Projections/LPPtoGC/Attributes/Edge/syn_id,\
/Projections/NGFCtoGC/Attributes/Edge/syn_id,\
/Projections/AACtoGC/Connectivity/Source\ Index,\
/Projections/BCtoGC/Connectivity/Source\ Index,\
/Projections/HCtoGC/Connectivity/Source\ Index,\
/Projections/HCCtoGC/Connectivity/Source\ Index,\
/Projections/MCtoGC/Connectivity/Source\ Index,\
/Projections/MOPPtoGC/Connectivity/Source\ Index,\
#/Projections/LPPtoGC/Connectivity/Source\ Index,\
/Projections/MPPtoGC/Connectivity/Source\ Index\
:GZIP=9 \
 -i $input -o $output




