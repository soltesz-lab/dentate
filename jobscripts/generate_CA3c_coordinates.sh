#!/bin/bash


ibrun -np 4 python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Ext.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       --template-path=./templates \
       -i CA3c \
       --output-path=./datasets/dentate_CA3c_coords_20190515.h5 \
       --output-namespace='Generated Coordinates' 




