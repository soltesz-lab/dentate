
module load python3
module load phdf5

set -x

export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

ibrun -np 4 python3 ./scripts/measure_trees.py \
    --config=./config/Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates \
    -i GC \
    --forest-path=$SCRATCH/striped/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_proximal_pf_20210114.h5  \
    --output-path=$SCRATCH/dentate/results/measure_trees.h5 \
    --io-size=1 -v

