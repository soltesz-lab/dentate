
#!/bin/bash

python ./scripts/generate_DG_PP_features_v3.py \
       --config='./config/Full_Scale_Control.yaml' \
       --coords-path='./datasets/DG_coords_20180717.h5' \
       --template-path='./templates' \
       --stimulus-id=100 \
       --output-path='./datasets/DG_PP_features_20180926.h5' \
       -v

