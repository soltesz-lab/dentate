#!/bin/bash

srcdir=$1
prefix=$2
forests="$3"

for forest in $forests; do

    echo forest = $forest
    mkdir -p $prefix/$forest
    src=${srcdir}/${forest}
    cat ${src}/MPPtoDGC*.dat > $prefix/$forest/MPPtoDGC.dat

done
