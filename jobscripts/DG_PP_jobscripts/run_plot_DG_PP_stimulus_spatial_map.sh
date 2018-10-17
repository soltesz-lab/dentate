
#!/bin/bash

mpirun -n 1 python plot_stimulus_spatial_map.py --features-path='DG_PP_spikes.h5' --features-namespace='Vector Stimulus 100' --coords-path='../config/DG_coords_20180717.h5' --distances-namespace='Arc Distances' --include='MPP' --trajectory-id='100' --bin-size=100 --from-spikes=True

mpirun -n 1 python plot_stimulus_spatial_map.py --features-path='DG_PP_spikes.h5' --features-namespace='Vector Stimulus 100' --coords-path='../config/DG_coords_20180717.h5' --distances-namespace='Arc Distances' --include='LPP' --trajectory-id='100' --bin-size=100 --from-spikes=True

