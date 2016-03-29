#!/bin/bash

prefix=$1
output=$2
forests="$3"

for forest in $forests; do

    echo forest = $forest
    sort -n $prefix/$forest/LPPtoDGCtargets.dat | uniq -c >> $output/histLPPtoDGCtargets.dat

done
