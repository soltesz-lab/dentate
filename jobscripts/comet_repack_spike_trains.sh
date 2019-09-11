#!/bin/bash
#
#SBATCH -J repack_spike_trains
#SBATCH -o ./results/repack_spike_trains.%j.o
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
export input=$prefix/DG_input_spike_trains_20190909.h5
export output=$prefix/DG_input_spike_trains_20190909_compressed.h5

##h5copy -v -i $input -o $copy -s /Populations -d /Populations

h5repack -L -v -f SHUF -f GZIP=9 \
 -i $input -o $output




