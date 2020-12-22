#!/bin/bash -l
export DATE=$(date +%Y%m%d_%H%M%S)
export JOB_NAME=troubleshoot_optimize_DG_test_network_subworlds_dbg_"$DATE"
sbatch <<EOT
#!/bin/bash -l
#SBATCH -J $JOB_NAME
#SBATCH -o /scratch1/06441/aaronmil/logs/dentate/$JOB_NAME.%j.o
#SBATCH -e /scratch1/06441/aaronmil/logs/dentate/$JOB_NAME.%j.e
#SBATCH -p development
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH -t 2:00:00
#SBATCH --mail-user=neurosutras@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

set -x

cd $WORK/dentate

ibrun -n 2240 python3 -m nested.optimize \
  --config-file-path=config/troubleshoot_DG_optimize_network_subworlds_config_dbg.yaml \
  --output-dir=$SCRATCH/data/dentate/results \
  --framework=pc \
  --verbose \
  --procs_per_worker=112 \
  --no_cleanup \
  --param_config_name="Weight exc inh microcircuit" \
  --arena_id=A --trajectory_id=Diag \
  --template_paths=../DGC/Mateos-Aparicio2014:templates \
  --dataset_prefix=/scratch1/03320/iraikov/striped/dentate \
  --config_prefix=config \
  --results_path=$SCRATCH/data/dentate/results \
  --spike_input_path="/scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_spike_trains_20200910_compressed.h5" \
  --spike_input_namespace='Input Spikes A Diag' \
  --spike_input_attr='Spike Train' \
  --target_rate_map_path="/scratch1/03320/iraikov/striped/dentate/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
  --target_rate_map_namespace="Place Selectivity" \
  --max_walltime_hours=2.0 \
  --io_size=8 \
  --microcircuit_inputs \
  --t_start=0. \
  --tstop=10. \
  --pop_size=20 \
  --path_length=1 \
  --max_iter=1
EOT
