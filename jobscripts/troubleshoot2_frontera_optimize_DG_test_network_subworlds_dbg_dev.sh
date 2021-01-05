#!/bin/bash -l
#SBATCH -J troubleshoot2_frontera_optimize_DG_test_network_subworlds
#SBATCH -o ./results/troubleshoot2_frontera_optimize_DG_test_network_subworlds.%j.o
#SBATCH -e ./results/troubleshoot2_frontera_optimize_DG_test_network_subworlds.%j.e
#SBATCH -p development
#SBATCH -N 40
#SBATCH -n 2240
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL


module load phdf5
set -x

export MPLBACKEND=PS
export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

cd $WORK/dentate


mpirun -n 10 python3 -m nested.optimize \
  --config-file-path=$DG_HOME/config/troubleshoot2_DG_optimize_network_subworlds_config_dbg.yaml \
  --output-dir=$SCRATCH/dentate/results \
  --framework=pc \
  --verbose=True \
  --debug \
  --disp \
  --procs_per_worker=10 \
  --no_cleanup \
  --param_config_name="Weight exc inh microcircuit" \
  --arena_id=A --trajectory_id=Diag \
  --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
  --dataset_prefix=/scratch1/03320/iraikov/striped/dentate \
  --config_prefix=$DG_HOME/config \
  --results_path=$SCRATCH/dentate/results \
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
  --pop_size=1 \
  --path_length=1 \
  --max_iter=1

