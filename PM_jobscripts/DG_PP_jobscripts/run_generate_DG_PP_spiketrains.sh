
#!/bin/bash

mpirun -n 1 python generate_DG_PP_spiketrains.py --config="../config/Full_Scale_Control.yaml" --features-path="DG_PP_features.h5" --stimulus-id=100 --template-path='../templates' --output-path="DG_PP_spikes_test.h5" -v
