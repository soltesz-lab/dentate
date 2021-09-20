
mpirun.mpich -n 8 python3 ./scripts/distribute_synapse_locs.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --template-path=templates:$HOME/src/model/DGC/Mateos-Aparicio2014 \
             --populations=GC \
             --forest-path=./datasets/Slice/dentatenet_Full_Scale_GC_Aradi_Sat_SLN_proximal_pf_20210915.h5 \
             --output-path=./datasets/Slice/dentatenet_Full_Scale_GC_Aradi_Sat_SLN_proximal_pf_20210916.h5 \
             --distribution=poisson \
             --io-size=2 -v 


