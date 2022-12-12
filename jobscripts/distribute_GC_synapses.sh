
mpirun.mpich -n 1 python3 ./scripts/distribute_synapse_locs.py \
             --config=Full_Scale_Basis.yaml \
             --config-prefix=./config \
             --template-path=templates \
             --populations=GC \
             --forest-path=./datasets/Slice/dentatenet_Full_Scale_GC_SLN_proximal_pf_20220919.h5 \
             --output-path=./datasets/Slice/dentatenet_Full_Scale_GC_SLN_proximal_pf_20220919.h5 \
             --distribution=poisson \
             --io-size=1 -v


