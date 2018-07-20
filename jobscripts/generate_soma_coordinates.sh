#!/bin/bash


mpirun -np 4 -bind-to none python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Control.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       -i AAC -i BC -i MC -i HC -i HCC -i IS -i MOPP -i NGFC -i MPP -i LPP \
       --output-path=./datasets/dentate_Full_Scale_Control_coords_20180717.h5 \
       --output-namespace='Generated Coordinates' 




