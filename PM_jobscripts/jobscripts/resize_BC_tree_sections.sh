#!/bin/bash


python3 ./scripts/resize_tree_sections.py \
    --population=BC \
    --config-prefix=$HOME/src/model/dentate/config \
    --config=Full_Scale_Basis.yaml \
    --forest-path=./datasets/Small_Scale/BC_axon_20200713.h5 \
    --output-path=./datasets/Small_Scale/BC_axon_20200723.h5 \
    -v



