#!/bin/bash


mpirun -np 8 python ./scripts/generate_soma_coordinates.py -v \
       --config=./config/Full_Scale_Control.yaml \
       --types-path=./datasets/dentate_h5types.h5 \
       --output-path=./datasets/dentate_Full_Scale_Control_coords_20180709.h5 \
       -i AAC -i BC -i MC -i HC -i HCC -i IS -i MOPP -i NGFC -i MPP -i LPP \
       --output-namespace='Generated Coordinates' 



