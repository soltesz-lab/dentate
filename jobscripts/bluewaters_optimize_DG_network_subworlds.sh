### set the number of nodes and the number of PEs per node
#PBS -l nodes=1600:ppn=32:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=12:00:00
### set the job name
#PBS -N optimize_DG_network_subworlds
### set the job stdout and stderr
#PBS -e ./results/optimize_DG_network_subworlds.$PBS_JOBID.err
#PBS -o ./results/optimize_DG_network_subworlds.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A bayj


module load bwpy/2.0.1

set -x

export LC_ALL=en_IE.utf8
export LANG=en_IE.utf8
export SCRATCH=/projects/sciteam/bayj
export NEURONROOT=$SCRATCH/nrnintel3
export PYTHONPATH=$HOME/model:$HOME/model/dentate:$NEURONROOT/lib/python:$SCRATCH/site-packages:$PYTHONPATH
export PATH=$NEURONROOT/x86_64/bin:$PATH
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

results_path=$DG_HOME/results/optimize_DG_network_subworlds.$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

aprun -n 12800 -N 8 -d 4 -b -- bwpy-environ -- \
    python3.6 -m nested.optimize  \
    --config-file-path=$DG_HOME/config/DG_optimize_network_subworlds_config.yaml \
    --output-dir=$results_path \
    --pop_size=32 \
    --max_iter=5 \
    --path_length=1 \
    --framework=pc \
    --disp \
    --verbose \
    --procs_per_worker=400 \
    --no_cleanup \
    --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
    --dataset_prefix="$SCRATCH" \
    --config_prefix=$DG_HOME/config \
    --results_path=$results_path \
    --cell_selection_path=$DG_HOME/datasets/DG_slice_20190729.yaml \
    --spike_input_path="$SCRATCH/Full_Scale_Control/DG_input_spike_trains_20190724_compressed.h5" \
    --spike_input_namespace='Input Spikes A Diag' \
    --spike-input-attr='Spike Train' \
    --max_walltime_hours=11.75 \
    --io_size=24 \
    -v
