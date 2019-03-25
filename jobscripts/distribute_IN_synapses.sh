
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=Full_Scale_Pas.yaml \
               --config-prefix=./config \
               --template-path=./templates \
               -i AAC -i BC -i MC -i IS -i HC -i HCC -i MOPP -i NGFC \
               --forest-path=./datasets/DG_IN_forest_20190325.h5 \
               --output-path=./datasets/DG_IN_forest_syns_20190325.h5 \
               --io-size=4 -v
              

##               --forest-path=./datasets/DG_IN_forest_20180908.h5 \
##               --output-path=./datasets/DG_IN_forest_syns_20180908.h5  \
##
