#!/bin/bash
#
#SBATCH -J repack_GC_connections
#SBATCH -o ./results/repack_GC_connections.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH -p shared
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x
export prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control
export copy=$prefix/DG_GC_connections_20171105.h5
export output=$prefix/DG_GC_connections_20171105_compressed.h5

##h5copy -s /Projections -d /Projections -i $input -o $copy
h5repack -v -f GZIP=9 -i $copy -o $output




