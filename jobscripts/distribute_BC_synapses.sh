
mpirun.mpich -np 1 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=Full_Scale_Pas.yaml \
               --config-prefix=./config \
               --template-path=$PWD/templates \
               -i BC \
               --forest-path=./datasets/BC_forest_20181226.h5 \
               --output-path=./datasets/BC_forest_syns_20190123.h5  \
               --io-size=1 -v 
              
