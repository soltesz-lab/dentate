
#!/bin/bash

python generate_DG_PP_features_v2.py --config='../config/Full_Scale_Control.yaml' --coords-path='../config/DG_coords_20180717.h5' --template-path='../templates' --stimulus-id=100 -v --output-path='test.h5'
