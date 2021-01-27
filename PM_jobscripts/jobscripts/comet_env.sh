#!/bin/bash

module load python
module load hdf5


export PYTHONPATH=$HOME/.local/lib/python3.6/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/$USER/temp_project

export MPIRUN=ibrun

