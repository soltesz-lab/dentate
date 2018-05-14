
mpirun.mpich -np 8 python ./scripts/measure_distances.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates \
              -i MC -i HC -i HCC -i NGFC -i MOPP -i IS -i LPP -i MPP \
              --coords-path=./datasets/DG_coords_20180507.h5 \
              --rotate=-35 \
              --io-size=2 -v
##              -i AAC -i BC  \

