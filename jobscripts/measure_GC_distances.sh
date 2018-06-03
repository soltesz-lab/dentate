
mpirun.mpich -np 8 python ./scripts/measure_distances.py \
             --config=./config/Full_Scale_Control.yaml \
             --coords-namespace=Coordinates \
             -i GC \
              --coords-path=./datasets/DG_coords_20180602.h5 \
              --io-size=2 --interpolate -v 

