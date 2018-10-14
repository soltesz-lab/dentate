import h5py

h5types_file = 'dentate_h5types.h5'

DG_populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP"]
DG_IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
DG_EXT_populations = ["MPP", "LPP"]

DG_cells_file = "DG_Cells_Full_Scale_20181012.h5"
DG_connections_file = "DG_Connections_Full_Scale_20181012.h5"

DG_GC_coordinate_file  = "DGC_forest_reindex_20180418.h5"
DG_IN_coordinate_file  = "dentate_Full_Scale_Control_coords_20180717.h5"
DG_EXT_coordinate_file = "dentate_Full_Scale_Control_coords_20180717.h5"

DG_GC_forest_file = "DGC_forest_20180425.h5"
DG_IN_forest_file = "DG_IN_forest_20180908.h5"

DG_GC_forest_syns_file = "DGC_forest_syns_20180812_compressed.h5"
DG_IN_forest_syns_file = "DG_IN_forest_syns_20180908.h5"

DG_GC_syn_weights_file = "DG_GC_forest_syns_log_normal_weights_20181010_compressed.h5"
DG_IN_syn_weights_file = "DG_IN_syns_log_normal_weights_20181011_compressed.h5"

DG_IN_connectivity_file = "DG_IN_connections_20180908.h5"
DG_GC_connectivity_file = "DG_GC_connections_20180813_compressed.h5"

DG_vecstim_file_dict = { 
    100: "DG_PP_spiketrains_100_20180928.h5", 
}

vecstim_dict = { 'Vector Stimulus %i' % stim_id : stim_file for stim_id, stim_file in DG_vecstim_file_dict.items() }
     


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
     'MPP':  DG_EXT_coordinate_file,
     'LPP':  DG_EXT_coordinate_file
}

coordinate_ns_generated = 'Generated Coordinates'
coordinate_ns_interpolated = 'Interpolated Coordinates'
coordinate_namespaces = {
     'AAC':  coordinate_ns_generated,
     'BC':   coordinate_ns_generated,
     'GC':   coordinate_ns_interpolated,
     'HC':   coordinate_ns_generated,
     'HCC':  coordinate_ns_generated,
     'IS':   coordinate_ns_generated,
     'MC':   coordinate_ns_generated,
     'MOPP': coordinate_ns_generated,
     'NGFC': coordinate_ns_generated,
     'MPP':  coordinate_ns_generated,
     'LPP':  coordinate_ns_generated
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
     'GC': { "Log-Normal Weights": DG_GC_syn_weights_file },
     'MC': { "Log-Normal Weights": DG_IN_syn_weights_file } 
}

## Creates H5Types entries
with h5py.File(DG_cells_file) as f:
    input_file  = h5py.File(h5types_file,'r')
    input_file.copy('/H5Types',f)
    input_file.close()
with h5py.File(DG_connections_file) as f:
    input_file  = h5py.File(h5types_file,'r')
    input_file.copy('/H5Types',f)
    input_file.close()

## Creates coordinates entries
with h5py.File(DG_cells_file) as f:

    grp = f.create_group("Populations")
                
    for p in DG_populations:
        grp.create_group(p)

    for p in DG_populations:
        coords_file = coordinate_files[p]
        coords_ns   = coordinate_namespaces[p]
        grp[p]["Coordinates"] = h5py.ExternalLink(coords_file,"/Populations/%s/%s" % (p, coords_ns))

## Creates forest entries and synapse attributes
with h5py.File(DG_cells_file) as f:

    grp = f["Populations"]

    for p in DG_populations:
        if p in forest_files:
            forest_file = forest_files[p]
            forest_syns_file = forest_syns_files[p]
            grp[p]["Trees"] = h5py.ExternalLink(forest_file,"/Populations/%s/Trees" % p)
            grp[p]["Synapse Attributes"] = h5py.ExternalLink(forest_syns_file,"/Populations/%s/Synapse Attributes" % p)

    for p in DG_populations:
        if p in syn_weight_files:
            weight_dict = syn_weight_files[p]
            for w in weight_dict:
                grp[p][w] = h5py.ExternalLink(weight_dict[w],"/Populations/%s/%s" % (p,w))

## Creates connectivity entries
with h5py.File(DG_connections_file) as f:

    grp = f.create_group("Projections")
               
    for p in DG_IN_populations:
        grp[p] = h5py.ExternalLink(DG_IN_connectivity_file,"/Projections/%s" % p)
    
    grp['GC'] = h5py.ExternalLink(DG_GC_connectivity_file,"/Projections/%s" % 'GC')

## Creates vector stimulus entries
with h5py.File(DG_cells_file) as f:

    grp = f["Populations"]

    for p in DG_EXT_populations:
        for (vecstim_ns, vecstim_file) in vecstim_dict.items():
            grp[p][vecstim_ns] = h5py.ExternalLink(vecstim_file,"/Populations/%s/%s" % (p, vecstim_ns))


    
