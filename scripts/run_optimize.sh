
#!/bin/bash

template='dentate_h5types.h5'
output='MPP.h5'
output_temp='temp-MPP.h5'
threads=$1

#cp $template $output
#cp $template $output_temp

echo "Running optimization routine..."
mpirun.mpich -n $threads python generate_DG_PP_features_reduced_h5support.py --types-path=$template --output-path=$output -o --verbose --lbound=1 --ubound=100 
#echo "..complete... generating data output for visualization"
#mpiexec -n 1 python generate_DG_PP_data.py $output_temp
echo "...complete..."

