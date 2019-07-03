
mpirun.mpich -np 8 python3 ./scripts/distribute_synapse_locs.py \
               --distribution=poisson \
               --config=Full_Scale_Basis.yaml \
               --config-prefix=./config \
               --template-path=./templates \
               -i AAC -i BC -i MC -i IS -i HC -i HCC -i MOPP -i NGFC \
               --forest-path=./datasets/Test_GC_1000/DG_Test_GC_1000_forest_20190612.h5 \
               --output-path=./datasets/Test_GC_1000/DG_Test_GC_1000_cells_20190612.h5 \
               --io-size=4 -v
