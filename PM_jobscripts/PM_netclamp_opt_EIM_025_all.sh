ml load intel19
export FI_MLX_ENABLE_SPAWN=yes
now=$(date +"%Y%m%d_%H%M%S")
Izhiseries='_Izhi_20201022_'
expfx=$now$Izhiseries'netclamp_opt_EIM_025'
ext='.log'
logdir=results/netclamp/logs20210128/
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_AAC.sh     2>&1 | tee $logdir/$expfx'AAC'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_BC.sh      2>&1 | tee $logdir/$expfx'BC'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_HICAP.sh   2>&1 | tee $logdir/$expfx'HICAP'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_HIPP.sh    2>&1 | tee $logdir/$expfx'HIPP'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_IS.sh      2>&1 | tee $logdir/$expfx'IS'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_MOPP.sh    2>&1 | tee $logdir/$expfx'MOPP'$ext  &
ibrun -n 8 ./PM_jobscripts/netclamp_opt_EIM_025_NGFC.sh    2>&1 | tee $logdir/$expfx'NGFC'$ext  

#ibrun -n 8 ./jobscripts/netclamp_opt_MC.sh*
