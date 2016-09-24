#!/bin/bash

prefix=$1
output=$2
forests="$3"

for forest in $forests; do

    echo forest = $forest
    sort -n $prefix/$forest/PPtoDGCtargets.dat | uniq -c >> $output/histPPtoDGCtargets.dat
    sort -n $prefix/$forest/PPtoDGCsources.dat | uniq -c >> $output/histPPtoDGCsources.dat

done
