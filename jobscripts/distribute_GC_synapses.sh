
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/src/model/DGC/Mateos-Aparicio2014 --populations=GC \
              --forest-path=./datasets/DGC_forest_test_syns_20180214.h5 \
              --distribution=poisson \
              --io-size=2

