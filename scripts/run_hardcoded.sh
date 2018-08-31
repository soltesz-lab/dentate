
#!/bin/bash

template='dentate_h5types.h5'
output='MPP.h5'
output_temp='temp-MPP.h5'
threads=$1

#cp dentate_h5types.h5 $output
#cp dentate_h5types.h5 $output_temp

echo "Generating cells and their features.."
mv $template ../
rm *.h5
mv ../$template .
mpirun.mpich -n $threads python generate_DG_PP_features_reduced_h5support.py --types-path=$template --output-path=$output --verbose

