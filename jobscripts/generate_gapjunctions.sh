#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_gapjunctions.py \
             --config=./config/Full_Scale_Pas_GJ.yaml \
             --types-path=./datasets/dentate_h5types_gj.h5 \
             --forest-path=./datasets/DG_IN_forest_20190325.h5 \
             --connectivity-path=./datasets/DG_gapjunctions_20190424.h5 \
             --connectivity-namespace="Gap Junctions" \
             --coords-path=./datasets/DG_coords_20190122.h5 \
             --coords-namespace="Coordinates" \
             --io-size=2 -v
