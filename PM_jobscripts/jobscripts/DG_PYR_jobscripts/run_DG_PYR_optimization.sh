#!/bin/bash

DENTATE_DIR=$HOME/soltesz-lab/dentate
SCRIPT_DIR=$DENTATE_DIR/scripts
CONFIG_DIR=$DENTATE_DIR/config

mpirun -n 3 python -m nested.optimize --config-file-path=$CONFIG_DIR/optimize_DG_PYR_backprojection.yaml --pop-size=12 --max-iter=50 --path-length=3 --disp --output-dir='data' --label=0

