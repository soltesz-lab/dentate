#!/bin/bash

if [ $# -eq 0 ]
  then
    start=0
    stop=9
    echo "No arguments supplied. Will loop over modules [$start,$stop]"
elif [$# -eq 0]
  then
    start=$1
    stop=$1
    echo "Will optimize over module $i"
elif [$# -eq 2]
  then
    start=$1
    stop=$2
    echo "Arguments supplied. Will loop over modules [$start,$stop]"
else
    echo "More than 2 arguments supplied, exiting..."
    exit 1
fi


config_directory='../config/'
output_dir='place_data_09032018_2/'
config_file_name='optimize_DG_PP_config_place_module_'
config_suffix='.yaml'
storage_suffix='.hdf5'

workers=4
path_length=3
max_iter=50
pop_size=48

for i in `seq $start $stop`;
do
    config_file_path=$config_directory$config_file_name$i$config_suffix
    storage_file_path=$output_dir$config_file_name$i$storage_suffix
    echo 'Running place cell module '$i' optimization'
    mpirun -n $workers python -m nested.optimize --config-file-path=$config_file_path --pop-size=$pop_size --max-iter=$max_iter --path-length=$path_length --disp --output-dir=$output_dir --storage-file-path=$storage_file_path
    
done
