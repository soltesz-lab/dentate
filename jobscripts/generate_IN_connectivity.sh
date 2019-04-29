#!/bin/bash

mpirun.mpich -np 8 python ./scripts/generate_distance_connections.py \
    --config=Full_Scale_Control.yaml \
    --forest-path=./datasets/DG_IN_forest_syns_20180908_compressed.h5 \
    --connectivity-path=./datasets/DG_IN_connections_20190426.h5 \
    --connectivity-namespace=Connections \
    --coords-path=./datasets/DG_coords_20180717.h5 \
    --coords-namespace="Coordinates" \
    --io-size=2 -v
