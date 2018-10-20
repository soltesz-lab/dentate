
#!/bin/bash

mpirun -n 1 python -i plot_DG_PP_metrics.py --coords-path='../config/DG_coords_20180717.h5' --features-path='DG_PP_features.h5' --distances-namespace='Arc Distances' --population='MPP' --cell-type='grid' --normed=0 --show-fig=1 --save-fig=1
