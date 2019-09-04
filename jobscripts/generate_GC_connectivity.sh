#!/bin/bash

mpirun.mpich -np 8 python3 ./scripts/generate_distance_connections.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --forest-path=./datasets/Test_GC_1000/DG_Test_GC_1000_cells_20190612.h5 \
             --connectivity-path=./datasets/Test_GC_1000/DG_Test_GC_1000_connections_20190612.h5 \
             --connectivity-namespace=Connections \
             --coords-path=./datasets/Test_GC_1000/DG_coords_20190521.h5 \
             --coords-namespace=Coordinates \
             --io-size=2 -v --dry-run
