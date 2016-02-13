#!/bin/bash

srcdir=$1
prefix=$2

for forest in `seq 1 1000`; do

    src=${srcdir}/PPprojection_Full_Scale_Control_forest_${forest}_*
    cat ${src}/PPtoDGCsources*.dat > $prefix/$forest/PPtoDGCsources.dat
    cat ${src}/PPtoDGCtargets*.dat > $prefix/$forest/PPtoDGCtargets.dat
    cat ${src}/PPtoDGCsections*.dat > $prefix/$forest/PPtoDGCsections.dat
    cat ${src}/PPtoDGCnodes*.dat > $prefix/$forest/PPtoDGCnodes.dat
    cat ${src}/PPtoDGClayers*.dat > $prefix/$forest/PPtoDGClayers.dat
    cat ${src}/PPtoDGCdistances*.dat > $prefix/$forest/PPtoDGCdistances.dat

done
