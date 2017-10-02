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
export input=$prefix/DGC_forest_20170509.h5
export copy=$prefix/DGC_forest_20170509_copy.h5
export output=$prefix/DGC_forest_20170509_compressed.h5

h5copy -v -i $input -o $copy -s /Populations -d /Populations

h5repack -L -v \
-f /Populations/GC/Trees/Destination\ Section,\
/Populations/GC/Trees/Parent\ Point,\
/Populations/GC/Trees/Point\ Layer,\
/Populations/GC/Trees/Radius,\
/Populations/GC/Trees/SWC\ Type,\
/Populations/GC/Trees/Section,\
/Populations/GC/Trees/Section\ Pointer,\
/Populations/GC/Trees/Source\ Section,\
/Populations/GC/Trees/Topology\ Pointer,\
/Populations/GC/Trees/Tree\ ID,\
/Populations/GC/Trees/X\ Coordinate,\
/Populations/GC/Trees/Y\ Coordinate,\
/Populations/GC/Trees/Z\ Coordinate:SHUF \
-f /Populations/GC/Trees/Destination\ Section,\
/Populations/GC/Trees/Parent\ Point,\
/Populations/GC/Trees/Point\ Layer,\
/Populations/GC/Trees/Radius,\
/Populations/GC/Trees/SWC\ Type,\
/Populations/GC/Trees/Section,\
/Populations/GC/Trees/Section\ Pointer,\
/Populations/GC/Trees/Source\ Section,\
/Populations/GC/Trees/Topology\ Pointer,\
/Populations/GC/Trees/Tree\ ID,\
/Populations/GC/Trees/X\ Coordinate,\
/Populations/GC/Trees/Y\ Coordinate,\
/Populations/GC/Trees/Z\ Coordinate:GZIP=9 \
 -i $copy -o $output




