#!/bin/bash
#
#SBATCH -J repack_forest_syns
#SBATCH -o ./results/repack_forest_syns.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -p shared
#SBATCH -t 8:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x
export prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/
export input=$prefix/DGC_forest_syns_20200628.h5
export output=$prefix/DGC_forest_syns_20200628_compressed.h5

h5repack -v -f SHUF -f GZIP=9 -i $input -o $output




