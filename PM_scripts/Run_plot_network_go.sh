BC=(
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210317_150923_network_clamp.optimize.BC_20210317_032406_01503844.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210317_150923_network_clamp.optimize.BC_20210317_032407_28135771.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210317_150923_network_clamp.optimize.BC_20210317_032406_93454042.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210317_150923_network_clamp.optimize.BC_20210317_032407_74865768.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_BC_20210317_150923_network_clamp.optimize.BC_20210317_032407_52357252.h5'
)
HC=(
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210317_152031_network_clamp.optimize.HC_20210317_032407_28682721.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210317_152031_network_clamp.optimize.HC_20210317_032407_15879716.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210317_152031_network_clamp.optimize.HC_20210317_032407_63599789.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210317_152031_network_clamp.optimize.HC_20210317_032407_53736785.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HC_20210317_152031_network_clamp.optimize.HC_20210317_032407_45419272.h5'
)
MOPP=(
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210317_153343_network_clamp.optimize.MOPP_20210317_032406_85763600.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210317_153343_network_clamp.optimize.MOPP_20210317_032407_29079471.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210317_153343_network_clamp.optimize.MOPP_20210317_032407_68839073.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210317_153343_network_clamp.optimize.MOPP_20210317_032407_45373570.h5'
'/scratch1/04119/pmoolcha/HDM/dentate/results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_MOPP_20210317_153343_network_clamp.optimize.MOPP_20210317_032407_31571230.h5'
)
N_cores=1
#for fil in ${BC[@]}
#    do
#        ibrun -n $N_cores -o $((counter * 56)) -rr python3 scripts/plot_network_clamp.py -p $fil -g 1039000 -g 1039950 -g 1040900 -g 1041850 -g 1042799 -v &
#    counter=$((counter + 1))
#    done
#wait
for fil in ${HC[@]}
    do
        ibrun -n $N_cores -o $((counter * 1)) -rr python3 scripts/plot_network_clamp.py -p $fil -g 1030000 -g 1032250 -g 1034500 -g 1036750 -g 1038999 -v &
    counter=$((counter + 1))
    done
wait
for fil in ${MOPP[@]}
    do
        ibrun -n $N_cores -o $((counter * 1)) -rr python3 scripts/plot_network_clamp.py -p $fil -g 1052650 -g 1053650 -g 1054650 -g 1055650 -g 1056649 -v &
    counter=$((counter + 1))
    done
wait
