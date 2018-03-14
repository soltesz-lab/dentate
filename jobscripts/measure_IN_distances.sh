
mpirun.mpich -np 4 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              -i AAC -i BC -i MC -i HC -i HCC -i NGFC -i MOPP \
              --coords-path=./datasets/dentate_Full_Scale_Control_coords_20180305.h5 \
              --io-size=2

