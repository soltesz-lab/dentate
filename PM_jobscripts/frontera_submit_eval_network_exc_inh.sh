#!/bin/bash
#

export job_name="eval_network_exc_inh_${1}"
export output_name="./results/eval_network_exc_inh_${1}.%j.o"

echo sbatch -J $job_name -o $output_name ./jobscripts/frontera_eval_network_.sh $1 
sbatch -J $job_name -o $output_name ./jobscripts/frontera_eval_network.sh $1 

