
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=Full_Scale_Pas.yaml \
               --config-prefix=./config \
               --template-path=./templates \
               -i MC \
               --forest-path=./datasets/DG_IN_forest_20190325.h5 \
               --output-path=./datasets/DG_MC_forest_syns_20190426.h5 \
               --io-size=4 -v

