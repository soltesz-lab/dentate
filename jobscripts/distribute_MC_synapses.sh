
for population in MC; do

mpirun.mpich -np 8 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$PWD/templates --populations=$population \
              --forest-path=./datasets/${population}_forest_20180715.h5 \
              --output-path=./datasets/${population}_forest_syns_20180715.h5 \
              --distribution=poisson --io-size=2 -v

done

