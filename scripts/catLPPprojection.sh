#!/bin/bash

srcdir=$1
prefix=$2
forests="$3"

for forest in $forests; do

    echo forest = $forest
    mkdir -p $prefix/$forest
    src=${srcdir}/LPPprojection_Full_Scale_Control_forest_${forest}_*
    cat ${src}/LPPtoDGCsources*.dat > $prefix/$forest/LPPtoDGCsources.dat
    cat ${src}/LPPtoDGCtargets*.dat > $prefix/$forest/LPPtoDGCtargets.dat
    cat ${src}/LPPtoDGCsections*.dat > $prefix/$forest/LPPtoDGCsections.dat
    cat ${src}/LPPtoDGCnodes*.dat > $prefix/$forest/LPPtoDGCnodes.dat
    cat ${src}/LPPtoDGClayers*.dat > $prefix/$forest/LPPtoDGClayers.dat
    cat ${src}/LPPtoDGCdistances*.dat > $prefix/$forest/LPPtoDGCdistances.dat

done
