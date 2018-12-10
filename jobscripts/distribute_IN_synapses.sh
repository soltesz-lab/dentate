
mpirun.mpich -np 1 python ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=./config/Full_Scale_Control.yaml \
               --template-path=$PWD/templates \
               -i MC -i BC \
               --forest-path=./datasets/GC_MC_BC_trees_20181126.h5 \
               --output-path=./datasets/GC_MC_BC_trees_20181126.h5 \
               --io-size=1 -v 
              

##               --forest-path=./datasets/DG_IN_forest_20180908.h5 \
##               --output-path=./datasets/DG_IN_forest_syns_20180908.h5  \
## -i AAC -i BC -i MC -i IS -i HC -i HCC -i MOPP -i NGFC 
