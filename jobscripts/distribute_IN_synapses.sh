
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=./config/Full_Scale_Control.yaml \
               --template-path=$PWD/templates \
               -i IS -i AAC -i BC -i HC -i HCC -i MC -i MOPP -i NGFC \
               --forest-path=./datasets/DG_IN_forest_20180717.h5 \
               --output-path=./datasets/DG_IN_forest_syns_20180717.h5 \
               --io-size=2 -v
              
