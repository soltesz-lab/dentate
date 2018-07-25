
#!/bin/bash

output='MPP_grid.h5'
output_temp='temp_MPP_grid.h5'

cp dentate_h5types.h5 $output
cp dentate_h5types.h5 $output_temp

echo "Generating cells and their features.."
mpiexec -n 1 python generate_DG_PP_features_reduced_h5support.py --coords-path='dentate_h5types.h5' --output-path=$output --verbose
echo "..complete... generating data output for visualization"
mpiexec -n 1 python generate_DG_PP_data.py $output_temp
echo "...complete..."

