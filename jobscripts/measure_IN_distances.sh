
mpirun.mpich -np 8 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates \
              -i LPP -i MPP -i MC -i HC -i HCC -i NGFC -i IS \
              --coords-path=./datasets/DG_coords_20180507.h5 \
              --io-size=2 -v
##              -i AAC -i BC -i MOPP 

