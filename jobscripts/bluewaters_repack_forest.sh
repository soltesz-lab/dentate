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
#PBS -e ./results/$PBS_JOBID.err
#PBS -o ./results/$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel

set -x

cd $PBS_O_WORKDIR

export prefix=/projects/sciteam/baef
export input=$prefix/DGC_forest_syns_020217.h5
export copy=$prefix/DGC_forest_syns_020217_copy.h5
export output=$prefix/DGC_forest_syns_020217_compressed.h5

aprun h5copy -v -i $input -o $copy -s /Populations -d /Populations

aprun h5repack -L -v \
-f /Populations/GC/Synapse_Attributes/layer/gid,\
/Populations/GC/Synapse_Attributes/layer/ptr,\
/Populations/GC/Synapse_Attributes/layer/value,\
/Populations/GC/Synapse_Attributes/section/gid,\
/Populations/GC/Synapse_Attributes/section/ptr,\
/Populations/GC/Synapse_Attributes/section/value,\
/Populations/GC/Synapse_Attributes/syn_id/gid,\
/Populations/GC/Synapse_Attributes/syn_id/ptr,\
/Populations/GC/Synapse_Attributes/syn_id/value,\
/Populations/GC/Synapse_Attributes/syn_locs/value,\
/Populations/GC/Synapse_Attributes/syn_locs/gid,\
/Populations/GC/Synapse_Attributes/syn_locs/ptr,\
/Populations/GC/Synapse_Attributes/syn_type/gid,\
/Populations/GC/Synapse_Attributes/syn_type/ptr,\
/Populations/GC/Synapse_Attributes/syn_type/value,\
/Populations/GC/Synapse_Attributes/swc_type/gid,\
/Populations/GC/Synapse_Attributes/swc_type/ptr,\
/Populations/GC/Synapse_Attributes/swc_type/value:GZIP=9 \
 -i $copy -o $output




