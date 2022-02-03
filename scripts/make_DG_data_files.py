import os, sys
import h5py, pathlib
from dentate.utils import viewitems

def h5_copy_dataset(f_src, f_dst, dset_path):
    print(f"Copying {dset_path} from {f_src} to {f_dst} ...")
    target_path = str(pathlib.Path(dset_path).parent)
    f_src.copy(f_src[dset_path], f_dst[target_path])

h5types_file = 'dentate_h5types.h5'

DG_populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP", "CA3c", "ConMC"]
DG_IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
DG_EXT_populations = ["MPP", "LPP", "CA3c", "ConMC"]

DG_cells_file = "DG_Cells_Full_Scale_20220201.h5"
DG_connections_file = "DG_Connections_Full_Scale_20220201.h5"

DG_GC_coordinate_file  = "DG_coords_20190717_compressed.h5"
DG_IN_coordinate_file  = "DG_coords_20190717_compressed.h5"
DG_EXT_coordinate_file = "DG_coords_20190717_compressed.h5"

DG_GC_forest_file = "DGC_forest_normalized_20200628_compressed.h5"
DG_IN_forest_file = "DG_IN_forest_syns_20211026.h5"

DG_GC_forest_syns_file = "DGC_forest_syns_20210920_compressed.h5"
DG_IN_forest_syns_file = "DG_IN_forest_syns_20211026.h5"

DG_GC_syn_weights_LN_file = "DG_GC_syn_weights_LN_20210920_compressed.h5"
DG_GC_syn_weights_S_file = "DG_GC_syn_weights_S_20220201_compressed.h5"
DG_MC_syn_weights_LN_file = "DG_MC_syn_weights_LN_20211116_compressed.h5"
DG_MC_syn_weights_S_file = "DG_MC_syn_weights_S_20220201_compressed.h5"

DG_GC_connectivity_file = "DG_GC_connections_20210920_compressed.h5"
DG_IN_connectivity_file = "DG_IN_connections_20211116_compressed.h5"

connectivity_files = {
    'AAC': DG_IN_connectivity_file,
    'BC': DG_IN_connectivity_file,
    'GC': DG_GC_connectivity_file,
    'HC': DG_IN_connectivity_file,
    'HCC': DG_IN_connectivity_file,
    'IS': DG_IN_connectivity_file,
    'MC': DG_IN_connectivity_file,
    'MOPP': DG_IN_connectivity_file,
    'NGFC': DG_IN_connectivity_file
}

DG_vecstim_file_dict = { 
    'A Diag': "DG_input_spike_trains_phasemod_20220201_compressed.h5"

}

vecstim_dict = {'Input Spikes %s' % stim_id : stim_file for stim_id, stim_file in viewitems(DG_vecstim_file_dict)}

DG_remap_vecstim_file_dict = None
#DG_remap_vecstim_file_dict = { 
#    'A Diag': "DG_remap_spike_trains_20191113_compressed.h5",
#}


coordinate_files = {
     'AAC':  DG_IN_coordinate_file,
     'BC':   DG_IN_coordinate_file,
     'GC':   DG_GC_coordinate_file,
     'HC':   DG_IN_coordinate_file,
     'HCC':  DG_IN_coordinate_file,
     'IS':   DG_IN_coordinate_file,
     'MC':   DG_IN_coordinate_file,
     'MOPP': DG_IN_coordinate_file,
     'NGFC': DG_IN_coordinate_file,
     'CA3c':  DG_EXT_coordinate_file,
     'MPP':  DG_EXT_coordinate_file,
     'LPP':  DG_EXT_coordinate_file,
     'ConMC':  DG_EXT_coordinate_file
}

distances_ns = 'Arc Distances'
coordinate_ns = 'Coordinates'
coordinate_ns_generated = 'Generated Coordinates'
coordinate_ns_interpolated = 'Interpolated Coordinates'
coordinate_namespaces = {
     'AAC':  coordinate_ns,
     'BC':   coordinate_ns,
     'GC':   coordinate_ns,
     'HC':   coordinate_ns,
     'HCC':  coordinate_ns,
     'IS':   coordinate_ns,
     'MC':   coordinate_ns,
     'MOPP': coordinate_ns,
     'NGFC': coordinate_ns,
     'CA3c':  coordinate_ns,
     'MPP':  coordinate_ns,
     'LPP':  coordinate_ns,
     'ConMC':  coordinate_ns
}
    


forest_files = {
     'AAC': DG_IN_forest_file,
     'BC': DG_IN_forest_file,
     'GC': DG_GC_forest_file,
     'HC': DG_IN_forest_file,
     'HCC': DG_IN_forest_file,
     'IS': DG_IN_forest_file,
     'MC': DG_IN_forest_file,
     'MOPP': DG_IN_forest_file,
     'NGFC': DG_IN_forest_file 
}

forest_syns_files = {
     'AAC': DG_IN_forest_syns_file,
     'BC': DG_IN_forest_syns_file,
     'GC': DG_GC_forest_syns_file,
     'HC': DG_IN_forest_syns_file,
     'HCC': DG_IN_forest_syns_file,
     'IS': DG_IN_forest_syns_file,
     'MC': DG_IN_forest_syns_file,
     'MOPP': DG_IN_forest_syns_file,
     'NGFC': DG_IN_forest_syns_file 
}

syn_weight_files = {
     'GC': { 
             "LTP Structured Weights A": DG_GC_syn_weights_S_file,
             "LTD Structured Weights A": DG_GC_syn_weights_S_file,
             "Log-Normal Weights": DG_GC_syn_weights_LN_file ,
             "Normal Weights": DG_GC_syn_weights_LN_file,
             "NS Structured Weights A": DG_GC_syn_weights_LN_file,
     },

     'MC': { 
             "LTP Structured Weights A": DG_MC_syn_weights_S_file,
             "LTD Structured Weights A": DG_MC_syn_weights_S_file,
             "Log-Normal Weights": DG_MC_syn_weights_LN_file,
             "Normal Weights": DG_MC_syn_weights_LN_file,
             "NS Structured Weights A": DG_GC_syn_weights_LN_file,
     }


}

## Creates H5Types entries
with h5py.File(DG_cells_file, 'w') as f:
    input_file  = h5py.File(h5types_file,'r')
    h5_copy_dataset(input_file, f, '/H5Types')
    input_file.close()

## Creates coordinates entries
with h5py.File(DG_cells_file, 'a') as f_dst:

    grp = f_dst.create_group("Populations")
                
    for p in DG_populations:
        grp.create_group(p)

    for p in DG_populations:
        coords_file = coordinate_files[p]
        coords_ns   = coordinate_namespaces[p]
        coords_dset_path = f"/Populations/{p}/{coords_ns}"
        distances_dset_path = f"/Populations/{p}/Arc Distances"
        with h5py.File(coords_file, 'r') as f_src:
            h5_copy_dataset(f_src, f_dst, coords_dset_path)
            h5_copy_dataset(f_src, f_dst, distances_dset_path)


## Creates forest entries and synapse attributes
for p in DG_populations:
    if p in forest_files:
        forest_file = forest_files[p]
        forest_syns_file = forest_syns_files[p]
        forest_dset_path = f"/Populations/{p}/Trees"
        forest_syns_dset_path = f"/Populations/{p}/Synapse Attributes"
        cmd = f"h5copy -p -s '{forest_dset_path}' -d '{forest_dset_path}' " \
              f"-i {forest_file} -o {DG_cells_file}"
        print(cmd)
        os.system(cmd)
        cmd = f"h5copy -p -s '{forest_syns_dset_path}' -d '{forest_syns_dset_path}' " \
              f"-i {forest_syns_file} -o {DG_cells_file}"
        print(cmd)
        os.system(cmd)

        if p in syn_weight_files:
            weight_dict = syn_weight_files[p]
            for w in weight_dict:
                if isinstance(weight_dict[w], tuple):
                    syn_weight_file = weight_dict[w][1]
                    syn_weight_dset_path = f"/Populations/{p}/{weight_dict[w][0]}"
                else:
                    syn_weight_file = weight_dict[w]
                    syn_weight_dset_path = f"/Populations/{p}/{w}"
                cmd = f"h5copy -p -s '{syn_weight_dset_path}' -d '{syn_weight_dset_path}' " \
                      f"-i {syn_weight_file} -o {DG_cells_file}"
                print(cmd)
                os.system(cmd)
                

## Creates vector stimulus entries
for (vecstim_ns, vecstim_file) in viewitems(vecstim_dict):
    for p in DG_EXT_populations+['GC']:
        vecstim_dset_path = f"/Populations/{p}/{vecstim_ns}"
        cmd = f"h5copy -p -s '{vecstim_dset_path}' -d '{vecstim_dset_path}' " \
              f"-i {vecstim_file} -o {DG_cells_file}"
        print(cmd)
        os.system(cmd)

    # if DG_remap_vecstim_file_dict is not None:
    #     for stim_id, vecstim_file in viewitems(DG_remap_vecstim_file_dict):
    #         vecstim_ns = 'Input Spikes %s' % stim_id
    #         remap_vecstim_ns = 'Input Spikes Remap %s' % stim_id
    #         for p in DG_EXT_populations:
    #             grp[p][remap_vecstim_ns] = h5py.ExternalLink(vecstim_file,"/Populations/%s/%s" % (p, vecstim_ns))


with h5py.File(DG_connections_file, 'w') as f:
    input_file  = h5py.File(h5types_file,'r')
    h5_copy_dataset(input_file, f, '/H5Types')
    input_file.close()

## Creates connectivity entries
for p in DG_populations:
     if p in connectivity_files:
         connectivity_file = connectivity_files[p]
         projection_dset_path = f"/Projections/{p}"
         cmd = f'h5copy -p -s {projection_dset_path} -d {projection_dset_path} ' \
               f'-i {connectivity_file} -o {DG_connections_file}'
         print(cmd)
         os.system(cmd)

