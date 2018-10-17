
#!/bin/bash

mpirun -n 1 python -i plot_DG_PP_features.py --features-path='DG_PP_features.h5' --cell-type='both' --show-fig=0 --save-fig=1
