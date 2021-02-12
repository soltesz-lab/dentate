
mpirun.mpich -np 4 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              -i GC --coords-namespace='Interpolated Coordinates' \
              --coords-path=./datasets/dentate_GC_coords_20180418.h5 \
              --io-size=2 -v

