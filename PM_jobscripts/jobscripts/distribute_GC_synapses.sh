
mpirun.mpich -np 8 python3 ./scripts/distribute_synapse_locs.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --template-path=templates:$HOME/src/model/DGC/Mateos-Aparicio2014 \
             --populations=GC \
             --forest-path=./datasets/Test_GC_1000/DG_Test_GC_1000_forest_20190612.h5 \
             --output-path=./datasets/Test_GC_1000/DG_Test_GC_1000_cells_20190612.h5 \
             --distribution=poisson \
             --io-size=2 --cache-size=10 -v 


