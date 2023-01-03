#!/bin/bash

mpirun.mpich -np 1 python3 ./scripts/generate_distance_connections.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --forest-path=./datasets/Slice/dentatenet_GC_AradiMorphNoTaper_proximal_pf_gid_118936_20220605.h5 \
             --connectivity-path=./datasets/Slice/dentatenet_GC_AradiMorphNoTaper_proximal_pf_gid_118936_20220605.h5 \
             --connectivity-namespace=Connections \
             --coords-path=./datasets/DG_coords_20190717_compressed.h5 \
             --coords-namespace=Coordinates \
             --io-size=1 -v
