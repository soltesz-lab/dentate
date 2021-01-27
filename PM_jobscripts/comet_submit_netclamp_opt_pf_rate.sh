#!/bin/bash
#

export job_name="netclamp_opt_pf_extent_features_${1}"
export output_name="./results/netclamp_opt_pf_extent_features_${1}.%j.o"

echo sbatch -J $job_name -o $output_name ./jobscripts/comet_netclamp_opt_pf_rate.sh $1
sbatch -J $job_name -o $output_name ./jobscripts/comet_netclamp_opt_pf_rate.sh $1

