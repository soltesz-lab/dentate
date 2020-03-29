#!/bin/bash
#
#SBATCH -J repack_GC_forest
#SBATCH -o ./results/repack_GC_forest.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=12
#SBATCH --mem=60G
#SBATCH -p shared
#SBATCH -t 6:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x
export prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/
export input=$prefix/DGC_forest_normalized_20200311.h5
export output=$prefix/DGC_forest_normalized_20200311_compressed.h5


##h5copy -v -i $input -o $copy -s /Populations -d /Populations

h5repack -L -v -f SHUF -f GZIP=9 \
 -i $input -o $output




