#!/bin/bash
#
#SBATCH -J repack_GC_forest_syns_wgts
#SBATCH -o ./results/repack_GC_forest_syns_wgts.%j.o
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=16
#SBATCH -p shared
#SBATCH -t 12:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load hdf5

set -x
export prefix=/oasis/scratch/comet/iraikov/temp_project/dentate/Full_Scale_Control/
#export input=$prefix/DG_GC_syn_weights_LN_20181227.h5
#export output=$prefix/DG_GC_syn_weights_LN_20181227_compressed.h5
export input=$prefix/DG_IN_syns_log_normal_weights_20190131.h5
export output=$prefix/DG_IN_syns_log_normal_weights_20190131_compressed.h5

h5repack -L -v -f SHUF -f GZIP=9 -i $input -o $output




