#!/bin/bash

mpirun.mpich -np 8 python3 ./scripts/generate_gapjunctions.py \
             --config=./config/Full_Scale_Basis.yaml \
             --types-path=./datasets/dentate_h5types_gj.h5 \
             --forest-path=./datasets/DG_IN_forest_20191112_compressed.h5 \
             --connectivity-path=./datasets/DG_gapjunctions_20191112.h5 \
             --connectivity-namespace="Gap Junctions" \
             --coords-path=./datasets/DG_coords_20190717_compressed.h5 \
             --coords-namespace="Coordinates" \
             --io-size=4 -v
