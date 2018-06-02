
mpirun.mpich -np 1 python ./scripts/measure_distances.py \
             --config=./config/Full_Scale_Control.yaml \
             --coords-namespace=Coordinates \
             -i GC \
              --coords-path=./datasets/DG_coords_20180530.h5 \
              --io-size=1 -v 

