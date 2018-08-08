
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/src/model/DGC/Mateos-Aparicio2014 --populations=GC \
              --forest-path=./datasets/DGC_forest_test_20180322.h5 \
              --output-path=./datasets/Test_GC_1000/DGC_forest_test_syns_20180808.h5 \
              --distribution=poisson \
              --io-size=2 -v --dry-run

