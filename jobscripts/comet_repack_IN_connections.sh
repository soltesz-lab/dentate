#!/bin/bash
#
#SBATCH -J repack_IN_connections
#SBATCH -o ./results/repack_IN_connections.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -p shared
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

export prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control
#export copy=$prefix/DG_Ext_IN_connections_20190128.h5
#export output=$prefix/DG_Ext_IN_connections_20190128_compressed.h5
export copy=$prefix/DG_IN_connections_20190430.h5
export output=$prefix/DG_IN_connections_20190430_compressed.h5

set -x

##h5copy -s /Projections -d /Projections -i $input -o $copy
h5repack -v -f SHUF -f GZIP=9 -i $copy -o $output




