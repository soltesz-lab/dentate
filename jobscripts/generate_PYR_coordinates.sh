#!/bin/bash


mpirun -np 4 -bind-to none python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Ext.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       --template-path=./templates \
       -i PYR \
       --output-path=./datasets/dentate_PYR_coords_20190122.h5 \
       --output-namespace='Generated Coordinates' 




