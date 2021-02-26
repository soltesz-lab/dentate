#!/bin/bash
#

export job_name="optimize_selectivity_GC_proximal_pf_${1}"
export output_name="./results/optimize_selectivity_GC_proximal_pf_${1}.%j.o"

echo sbatch -J $job_name -o $output_name ./jobscripts/frontera_optimize_selectivity_GC_proximal_pf.sh $1 "$2" "$3"
sbatch -J $job_name -o $output_name ./jobscripts/frontera_optimize_selectivity_GC_proximal_pf.sh $1 "$2" "$3"

