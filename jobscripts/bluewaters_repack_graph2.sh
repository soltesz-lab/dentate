#!/bin/bash
### set the number of nodes and the number of PEs per node
#PBS -l nodes=1:ppn=8:xe
### which queue to use
#PBS -q high
### set the wallclock time
#PBS -l walltime=12:00:00
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

export prefix=/projects/sciteam/baef/Full_Scale_Control
export input=$prefix/DG_GC_connections_20171020.h5
export output=$prefix/DG_GC_connections_20171020_compressed.h5

aprun h5repack -v -f "/Projections/GC/AAC/Connections/distance,/Projections/GC/AAC/Edges/Destination Block Index,/Projections/GC/AAC/Edges/Destination Block Pointer,/Projections/GC/AAC/Edges/Destination Pointer,/Projections/GC/AAC/Edges/Source Index,/Projections/GC/AAC/Synapses/syn_id,/Projections/GC/BC/Connections/distance,/Projections/GC/BC/Edges/Destination Block Index,/Projections/GC/BC/Edges/Destination Block Pointer,/Projections/GC/BC/Edges/Destination Pointer,/Projections/GC/BC/Edges/Source Index,/Projections/GC/BC/Synapses/syn_id,/Projections/GC/HC/Connections/distance,/Projections/GC/HC/Edges/Destination Block Index,/Projections/GC/HC/Edges/Destination Block Pointer,/Projections/GC/HC/Edges/Destination Pointer,/Projections/GC/HC/Edges/Source Index,/Projections/GC/HC/Synapses/syn_id,/Projections/GC/HCC/Connections/distance,/Projections/GC/HCC/Edges/Destination Block Index,/Projections/GC/HCC/Edges/Destination Block Pointer,/Projections/GC/HCC/Edges/Destination Pointer,/Projections/GC/HCC/Edges/Source Index,/Projections/GC/HCC/Synapses/syn_id,/Projections/GC/LPP/Connections/distance,/Projections/GC/LPP/Edges/Destination Block Index,/Projections/GC/LPP/Edges/Destination Block Pointer,/Projections/GC/LPP/Edges/Destination Pointer,/Projections/GC/LPP/Edges/Source Index,/Projections/GC/LPP/Synapses/syn_id,/Projections/GC/MC/Connections/distance,/Projections/GC/MC/Edges/Destination Block Index,/Projections/GC/MC/Edges/Destination Block Pointer,/Projections/GC/MC/Edges/Destination Pointer,/Projections/GC/MC/Edges/Source Index,/Projections/GC/MC/Synapses/syn_id,/Projections/GC/MOPP/Connections/distance,/Projections/GC/MOPP/Edges/Destination Block Index,/Projections/GC/MOPP/Edges/Destination Block Pointer,/Projections/GC/MOPP/Edges/Destination Pointer,/Projections/GC/MOPP/Edges/Source Index,/Projections/GC/MOPP/Synapses/syn_id,/Projections/GC/MPP/Connections/distance,/Projections/GC/MPP/Edges/Destination Block Index,/Projections/GC/MPP/Edges/Destination Block Pointer,/Projections/GC/MPP/Edges/Destination Pointer,/Projections/GC/MPP/Edges/Source Index,/Projections/GC/MPP/Synapses/syn_id,/Projections/GC/NGFC/Connections/distance,/Projections/GC/NGFC/Edges/Destination Block Index,/Projections/GC/NGFC/Edges/Destination Block Pointer,/Projections/GC/NGFC/Edges/Destination Pointer,/Projections/GC/NGFC/Edges/Source Index,/Projections/GC/NGFC/Synapses/syn_id":GZIP=9 -i $input -o $output




