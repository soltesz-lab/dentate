
for population in MC; do

mpirun.mpich -np 1 python ./scripts/distribute_synapse_locs.py \
              --config=./config/Full_Scale_Control.yaml \
              --template-path=$PWD/templates --populations=$population \
              --forest-path=./datasets/${population}_forest_20180630.h5 \
              --output-path=./datasets/${population}_forest_syns_20180630.h5 \
              --distribution=poisson --io-size=1 -v

done

