
#!/bin/bash

template='dentate_h5types.h5'
output='MPP_grid.h5'
output_temp='temp_MPP_grid.h5'

#cp $template $output
#cp $template $output_temp

echo "Running optimization routine..."
mpiexec -n 1 python generate_DG_PP_features_reduced_h5support.py --coords-path=$template --output-path=$output -o --iterations 2 --input-path=$output_temp --verbose --lbound=1 --ubound=100
echo "..complete... generating data output for visualization"
mpiexec -n 1 python generate_DG_PP_data.py $output_temp
echo "...complete..."

