
mpirun.mpich -np 8 python ./scripts/measure_distances.py \
             --config=./config/Full_Scale_Control.yaml \
             --coords-namespace=Coordinates \
             -i MC -i LPP -i MPP -i HC -i HCC -i NGFC -i IS -i AAC -i BC -i MOPP \
             --resolution 30 30 10 \
             --coords-path=./datasets/DG_coords_20180717.h5 \
             --io-size=2 -v


