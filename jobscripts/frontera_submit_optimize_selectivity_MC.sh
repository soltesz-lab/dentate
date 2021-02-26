#!/bin/bash
#

export job_name="optimize_selectivity_MC_${1}"
export output_name="./results/optimize_selectivity_MC_${1}.%j.o"

echo sbatch -J $job_name -o $output_name ./jobscripts/frontera_optimize_selectivity_MC.sh $1 "$2" "$3"
sbatch -J $job_name -o $output_name ./jobscripts/frontera_optimize_selectivity_MC.sh $1 "$2" "$3"

