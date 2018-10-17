
#!/bin/bash

POPULATION=$1

for i in `seq 1 10`;
do
    echo ' module ' $i ' population ' $POPULATION
    SAVEFILE='population-'$POPULATION'-module-'$i'.png'
    mpirun -n 1 python plot_stimulus_rate.py --features-path='DG_PP_spikes.h5' --features-namespace='Vector Stimulus' --trajectory-id='100' --include=$POPULATION --module=$i --show-fig=0 --save-fig=$SAVEFILE

done

echo 'population ' $POPULATION
SAVEFILE='population-'$POPULATION'.png'
mpirun -n 1 python plot_stimulus_rate.py --features-path='DG_PP_spikes.h5' --features-namespace='Vector Stimulus' --trajectory-id='100' --include $POPULATION --show-fig 0 --save-fig $SAVEFILE



