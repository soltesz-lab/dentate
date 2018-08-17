
mpirun.mpich -np 1 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=./config/Full_Scale_Control.yaml \
               --template-path=$PWD/templates \
               -i BC -i MC  \
               --forest-path=./datasets/MC_BC_trees_20180817.h5 \
               --output-path=./datasets/DG_IN_trees_20180817.h5  \
               --io-size=1 -v
              
# -i IS -i AAC -i BC -i HC -i HCC -i MC -i MOPP -i NGFC 
