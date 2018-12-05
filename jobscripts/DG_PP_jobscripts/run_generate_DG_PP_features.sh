
#!/bin/bash

python ./scripts/generate_DG_PP_features.py \
       --config='./config/Full_Scale_Control.yaml' \
       --coords-path='./datasets/DG_coords_20180717.h5' \
       --template-path='./templates' \
       --stimulus-id=100 -v \
       --output-path='./datasets/DG_PP_features_100_20181017.h5'
