#!/bin/bash

prefix=$1
output=$2
forests="$3"

for forest in $forests; do

    echo forest = $forest
    sort -n $prefix/$forest/MPPtoDGCtargets.dat | uniq -c >> $output/histMPPtoDGCtargets.dat
    sort -n $prefix/$forest/MPPtoDGCsources.dat | uniq -c >> $output/histMPPtoDGCsources.dat

done
