
python3 ./scripts/distribute_synapse_locs.py \
        --config=Full_Scale_Basis.yaml \
        --config-prefix=./config \
        --template-path=templates \
        --populations=GC \
        --forest-path=./datasets/Single/dentatenet_Slice_SLN_gid_513710_20230215.h5 \
        --output-path=./datasets/Single/dentatenet_Slice_SLN_gid_513710_20230215.h5 \
        --distribution=poisson \
        --io-size=1 -v


