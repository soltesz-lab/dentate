
#!/bin/bash

mpirun -n 1 python generate_DG_PP_features.py --config='../config/Full_Scale_Control.yaml' --coords-path='../config/DG_coords_20180717.h5' --template-path='../templates' --stimulus-id=100 -v --output-path='DG_PP_features.h5'
