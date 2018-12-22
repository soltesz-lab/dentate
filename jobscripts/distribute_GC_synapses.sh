
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=templates:$HOME/src/model/DGC/Mateos-Aparicio2014 --populations=GC \
              --forest-path=./datasets/GC_MC_BC_trees_20181126.h5 \
              --output-path=./datasets/GC_MC_BC_trees_20181221.h5 \
              --distribution=poisson \
              --io-size=2 --cache-size=10 -v

