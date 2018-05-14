
mpirun.mpich -np 1 python ./scripts/measure_trees.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$HOME/src/model/DGC/Mateos-Aparicio2014 \
              -i GC \
              --forest-path=./datasets/Test_GC_1000/DG_test_cells_meas_20180413.h5 \
              --io-size=2

