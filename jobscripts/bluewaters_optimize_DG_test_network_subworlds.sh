### set the number of nodes and the number of PEs per node
#PBS -l nodes=8:ppn=8:xe
### which queue to use
#PBS -q normal
### set the wallclock time
#PBS -l walltime=4:00:00
### set the job name
#PBS -N optimize_DG_test_network_subworlds
### set the job stdout and stderr
#PBS -e ./results/optimize_DG_test_network_subworlds.$PBS_JOBID.err
#PBS -o ./results/optimize_DG_test_network_subworlds.$PBS_JOBID.out
### set email notification
##PBS -m bea
### Set umask so users in my group can read job stdout and stderr files
#PBS -W umask=0027
#PBS -A baqc

module swap PrgEnv-cray PrgEnv-gnu
module load cray-hdf5-parallel
module load bwpy 
module load bwpy-mpi

set -x

export PYTHONPATH=$HOME/model:$HOME/model/dentate/btmorph:$HOME/bin/nrn/lib/python:/projects/sciteam/baqc/site-packages:$PYTHONPATH
export PATH=$HOME/bin/nrn/x86_64/bin:$PATH
export SCRATCH=/projects/sciteam/baqc
export MODEL_HOME=$HOME/model
export DG_HOME=$MODEL_HOME/dentate

results_path=$DG_HOME/results/optimize_DG_test_network_subworlds.$PBS_JOBID
export results_path

cd $PBS_O_WORKDIR

mkdir -p $results_path

cd tests

aprun -n 64 -b -- bwpy-environ -- \
    python2.7 -m nested.optimize  \
     --config-file-path=$DG_HOME/config/DG_test_network_subworlds_config.yaml \
     --output-dir=$results_path \
     --pop-size=2 \
     --max-iter=2 \
     --path-length=1 \
     --disp \
     --procs-per-worker=32 \
     --no-cleanup \
     --verbose \
     --template_paths=$MODEL_HOME/dgc/Mateos-Aparicio2014:$DG_HOME/templates \
     --dataset_prefix="$SCRATCH" \
     --config_prefix=$DG_HOME/config \
     --results_path=$results_path \
     --cell_selection_path=$DG_HOME/datasets/GC_subnet.yaml \
     --spike_input_path=$DG_HOME/results/Full_Scale_GC_Exc_Sat_LN_9533687.bw/dentatenet_Full_Scale_GC_Exc_Sat_LN_results.h5 \
     --spike_input_namespace='Spike Events'
