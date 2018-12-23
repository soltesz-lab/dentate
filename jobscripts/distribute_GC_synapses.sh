
mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
             --config=Full_Scale_Pas.yaml \
             --config-prefix=./config \
             --template-path=templates:$HOME/src/model/DGC/Mateos-Aparicio2014 --populations=GC \
             --forest-path=./datasets/DGC_forest_test_20181222.h5 \
             --output-path=./datasets/DGC_forest_test_syns_20181222.h5 \
             --distribution=poisson \
             --io-size=2 --cache-size=10 -v


