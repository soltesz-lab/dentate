netclamp_go=(
results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_IS_1049650_20201029_224540.h5
results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_NGFC_1044650_20201029_224540.h5
results/netclamp/dentatenet_Full_Scale_GC_Exc_Sat_SLN_IN_Izh_results_HCC_1043250_20201029_224540.h5
)

for fil in "${netclamp_go[@]}"
do
ibrun -n 1 python3 scripts/plot_network_clamp.py -p $fil 
done
