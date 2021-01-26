#!/bin/bash
#

export job_name="netclamp_opt_pf_rate_GC_${1}"
export output_name="./results/netclamp_opt_pf_rate_GC_${1}.%j.o"

echo sbatch -J $job_name -o $output_name ./jobscripts/frontera_netclamp_opt_pf_rate_GC.sh $1 "$2" "$3"
sbatch -J $job_name -o $output_name ./jobscripts/frontera_netclamp_opt_pf_rate_GC.sh $1 "$2" "$3"

