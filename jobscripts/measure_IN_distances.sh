
mpirun.mpich -np 8 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates \
              -i BC \
              --coords-path=./datasets/DG_coords_20180507.h5 \
              --rotate=-35 \
              --io-size=2 -v
##              -i AAC -i BC -i MC -i HC -i HCC -i NGFC -i MOPP \

