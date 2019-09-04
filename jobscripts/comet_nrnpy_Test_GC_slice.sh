#!/bin/bash
#
#SBATCH -J dentate_Test_GC_slice
#SBATCH -o ./results/dentate_Test_GC_slice.%j.o
#SBATCH --nodes=10
#SBATCH --ntasks-per-node=12
#SBATCH -p compute
#SBATCH -t 2:00:00
#SBATCH --mail-user=ivan.g.raikov@gmail.com
#SBATCH --mail-type=END
#

module load python
module unload intel
module load gnu
module load openmpi_ib
module load mkl
module load hdf5

set -x

export PYTHONPATH=$HOME/.local/lib/python3.5/site-packages:/opt/sdsc/lib
export PYTHONPATH=$HOME/bin/nrnpython3/lib/python:$PYTHONPATH
export PYTHONPATH=$HOME/model:$PYTHONPATH
export SCRATCH=/oasis/scratch/comet/iraikov/temp_project
ulimit -c unlimited

results_path=$SCRATCH/dentate/results/Test_GC_slice_$SLURM_JOB_ID
export results_path

mkdir -p $results_path

#git ls-files | tar -zcf ${results_path}/dentate.tgz --files-from=/dev/stdin
#git --git-dir=../dgc/.git ls-files | grep Mateos-Aparicio2014 | tar -C ../dgc -zcf ${results_path}/dgc.tgz --files-from=/dev/stdin

ibrun -np 120 gdb -q -batch -x $HOME/gdb_script --args python3 ./scripts/main.py \
    --arena-id=A --trajectory-id=Diag \
    --config-file=Full_Scale_GC_Exc_Sat_DD_LNN.yaml \
    --config-prefix=./config \
    --template-paths=../dgc/Mateos-Aparicio2014:templates \
    --dataset-prefix="$SCRATCH/dentate" \
    --results-path=$results_path \
    --io-size=24 \
    --tstop=10 \
    --v-init=-75 \
    --max-walltime-hours=1.75 \
    --cell-selection-path=./datasets/DG_slice_20190729.yaml \
    --write-selection \
    --spike-input-path="$SCRATCH/dentate/Full_Scale_Control/DG_input_spike_trains_20190724_compressed.h5" \
    --spike-input-namespace='Input Spikes' \
    --verbose

