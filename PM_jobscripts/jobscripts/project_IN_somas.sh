
mpirun.mpich -np 8 python ./scripts/project_somas.py \
              --config=./config/Full_Scale_Control.yaml \
              --coords-namespace=Coordinates -l 3.0 \
              -i BC \
              --coords-path=./datasets/DG_coords_20180516.h5 \
              --io-size=2 -v
## -i LPP -i MPP -i MC -i HC -i HCC -i NGFC -i IS -i BC -i MOPP \ \

