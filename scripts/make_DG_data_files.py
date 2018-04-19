import h5py
import itertools

h5types_file = 'dentate_h5types.h5'

DG_populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP"]
DG_IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
DG_EXT_populations = ["MPP", "LPP"]

DG_cells_file = "DG_Cells_Full_Scale_20180417.h5"
DG_connections_file = "DG_Connections_Full_Scale_20180417.h5"

DG_GC_coordinate_file  = "DGC_forest_reindex_compressed_20180224.h5"
DG_IN_coordinate_file  = "dentate_generated_coords_20180305.h5"
DG_EXT_coordinate_file = "dentate_generated_coords_20180305.h5"

DG_GC_forest_file = "DGC_forest_20180306.h5"
DG_IN_forest_file = "DG_IN_forest_syns_20180411.h5"

DG_GC_forest_syns_file = "DGC_forest_syns_compressed_20180306.h5"
DG_IN_forest_syns_file = "DG_IN_forest_syns_20180411.h5"

DG_IN_connectivity_file = "DG_IN_connections_20180417.h5"
DG_GC_connectivity_file = "DG_GC_connections_compressed_20180319.h5"

DG_vecstim_file_dict = { 
    100: "DG_PP_features_100_20180406.h5", \
    110: "DG_PP_features_110_20180404.h5", \
    120: "DG_PP_features_120_20180417.h5"  \
}

vecstim_dict = { 'Vector Stimulus %i' % stim_id : stim_file for stim_id, stim_file in DG_vecstim_file_dict.iteritems() }
     


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
        if forest_files.has_key(p):
            forest_file = forest_files[p]
            forest_syns_file = forest_syns_files[p]
            grp[p]["Trees"] = h5py.ExternalLink(forest_file,"/Populations/%s/Trees" % p)
            grp[p]["Synapse Attributes"] = h5py.ExternalLink(forest_syns_file,"/Populations/%s/Synapse Attributes" % p)

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
        for (vecstim_ns, vecstim_file) in vecstim_dict.iteritems():
            grp[p][vecstim_ns] = h5py.ExternalLink(vecstim_file,"/Populations/%s/%s" % (p, vecstim_ns))


    
