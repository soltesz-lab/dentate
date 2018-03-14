
mpirun.mpich -np 8 python ./scripts/measure_trees.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/src/model/DGC/Mateos-Aparicio2014 \
              -i GC \
              --forest-path=./datasets/Test_GC_1000/DGC_forest_test_syns_20171019.h5 \
              --io-size=2

