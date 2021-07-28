#!/bin/bash -l
#SBATCH -J frontera_optimize_DG_network_subworlds
#SBATCH -o ./results/frontera_optimize_DG_network_subworlds.%j.o
#SBATCH -e ./results/frontera_optimize_DG_network_subworlds.%j.e
#SBATCH -p development
#SBATCH -N 3
#SBATCH -n 168
#SBATCH -t 0:30:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=BEGIN,END,FAIL

module load python3
module load phdf5

set -x

export DATA_PREFIX="/tmp/optimize_DG_network"
export CDTools=/home1/apps/CDTools/1.1

export MPLBACKEND=PS
export NEURONROOT=$SCRATCH/bin/nrnpython3_intel19
export PYTHONPATH=$HOME/model:$NEURONROOT/lib/python:$SCRATCH/site-packages/intel19:$PYTHONPATH
export PATH=${CDTools}/bin:$NEURONROOT/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate


cd $SLURM_SUBMIT_DIR


distribute.bash ${SCRATCH}/dentate/optimize_DG_network

mpirun python3 -m nested.optimize \
  --config-file-path=$DG_HOME/config/DG_optimize_network_subworlds.yaml \
  --output-dir=$SCRATCH/dentate/results \
  --framework=pc \
  --verbose=True \
  --disp \
  --procs_per_worker=168 \
  --no_cleanup \
  --param_config_name="Weight exc inh microcircuit" \
  --arena_id=A --trajectory_id=Diag \
  --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
  --dataset_prefix="$DATA_PREFIX" \
  --config_prefix=$DG_HOME/config \
  --results_path=$SCRATCH/dentate/results \
  --spike_input_path="$DATA_PREFIX/Slice/dentatenet_Full_Scale_GC_Exc_Sat_SLN_selection_neg2000_neg1925um_phasemod_20210526_compressed.h5" \
  --spike_input_namespace='Input Spikes A Diag' \
  --spike_input_attr='Spike Train' \
  --target_rate_map_path="$DATA_PREFIX/Full_Scale_Control/DG_input_features_20200910_compressed.h5" \
  --target_rate_map_namespace="Place Selectivity" \
  --max_walltime_hours=2.0 \
  --t_start=0. \
  --io_size=1 \
  --microcircuit_inputs \
  --pop_size=1 \
  --path_length=1 \
  --max_iter=1

