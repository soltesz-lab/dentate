
module load intel/18.0.5
module load python3
module load phdf5

set -x

export LD_PRELOAD=/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_core.so:/opt/intel/compilers_and_libraries_2018.5.274/linux/mkl/lib/intel64_lin/libmkl_sequential.so
export NEURONROOT=$HOME/bin/nrnpython3
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

ibrun -np 4 python3 ./scripts/measure_trees.py \
    --config=./config/Full_Scale_Basis.yaml \
    --template-path=$HOME/model/dgc/Mateos-Aparicio2014:templates \
    -i GC \
    --forest-path=$SCRATCH/dentate/Slice/dentatenet_Full_Scale_GC_Exc_Sat_DD_SLN_extent_arena_margin_20200901_compressed.h5 \
    --io-size=1 -v

