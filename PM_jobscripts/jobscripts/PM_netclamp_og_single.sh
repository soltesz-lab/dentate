now=$(date +"%Y%m%d_%H%M%S")
Izhiseries='_Izhi_20201022_'
expfx=$now$Izhiseries'netclamp_opt_'
ext='.log'
logdir=results/netclamp/logs/
cell=$1
cellshort=$2
#ibrun -n 8 ./jobscripts/netclamp_opt_$cell.sh   2>&1 | tee $logdir/$expfx$cell$ext  
now='20201030_014610'
cd results/netclamp
optcellyaml=`ls *$cellshort*$now*.yaml`
cd -

echo $optcellyaml

now=$(date +"%Y%m%d_%H%M%S")
expfx=$now$Optseries$Izhiseries'netclamp_go_'
logdirexpfx=$logdir$expfx

cp jobscripts/PM_netclamp_og_single.sh $logdirexpfx$cell.sh 

#ibrun -n 1 ./jobscripts/PM_netclamp_go_$cell.sh  $optcellyaml 2>&1 | tee $logdirexpfx$cell$ext  
now='20201030_022458'
cd results/netclamp
gocellh5=`ls *$cellshort*$now*.h5`
cd -

echo $gocellh5

ibrun -n 1 python3 scripts/plot_network_clamp.py -p results/netclamp/$gocellh5 
