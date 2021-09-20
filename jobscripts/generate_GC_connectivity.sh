#!/bin/bash

mpirun.mpich -np 8 python3 ./scripts/generate_distance_connections.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --forest-path=./datasets/Slice/dentatenet_Full_Scale_GC_Aradi_Sat_SLN_proximal_pf_20210916.h5 \
             --connectivity-path=./datasets/Slice/dentatenet_Full_Scale_GC_Aradi_Sat_SLN_proximal_pf_20210916.h5 \
             --connectivity-namespace=Connections \
             --coords-path=./datasets/DG_coords_20190717_compressed.h5 \
             --coords-namespace=Coordinates \
             --io-size=2 -v
