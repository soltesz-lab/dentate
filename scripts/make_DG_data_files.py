import h5py
import itertools

DG_populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP"]
DG_IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
DG_EXT_populations = ["MPP", "LPP"]

DG_cells_file = "DG_Cells_Full_Scale_20180326.h5"
DG_connections_file = "DG_Connections_Full_Scale_20180326.h5"

DG_GC_coordinate_file  = "DGC_forest_reindex_compressed_20180224.h5"
DG_IN_coordinate_file  = "dentate_generated_coords_20180305.h5"
DG_EXT_coordinate_file = "dentate_generated_coords_20180305.h5"

DG_GC_forest_file = "DGC_forest_syns_compressed_20180306.h5"
DG_IN_forest_file = "DG_IN_forest_syns_20180304.h5"

DG_IN_connectivity_file = "DG_IN_connections_20180323.h5"
DG_GC_connectivity_file = "DG_GC_connections_compressed_20180319.h5"

DG_vecstim_file = "DG_PP_features_20180326.h5"

vecstim_dict = {
     'Vector Stimulus 0': DG_vecstim_file
}

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

## Creates coordinates entries
with h5py.File(DG_cells_file) as f:

    grp = f.create_group("Populations")
                
    for p in DG_populations:
        grp.create_group(p)

    for p in DG_populations:
        coords_file = coordinate_files[p]
        coords_ns   = coordinate_namespaces[p]
        grp[p]["Coordinates"] = h5py.ExternalLink(coords_file,"/Populations/%s/%s" % (p, coords_ns))

## Creates forest entries
with h5py.File(DG_cells_file) as f:

    grp = f["Populations"]

    for p in DG_populations:
        if forest_files.has_key(p):
            forest_file = forest_files[p]
            grp[p]["Trees"] = h5py.ExternalLink(forest_file,"/Populations/%s/Trees" % p)
            grp[p]["Synapse Attributes"] = h5py.ExternalLink(forest_file,"/Populations/%s/Synapse Attributes" % p)

## Creates connectivity entries
with h5py.File(DG_connections_file) as f:

    grp = f.create_group("Projections")
               
    for p in DG_IN_populations:
        grp[p] = h5py.ExternalLink(DG_IN_connectivity_file,"/Projections/%s" % p)
    
    grp['GC'] = h5py.ExternalLink(DG_GC_connectivity_file,"/Projections/%s" % 'GC')

with h5py.File(DG_cells_file) as f:

    grp = f["Populations"]

    for p in DG_EXT_populations:
        for (vecstim_ns, vecstim_file) in vecstim_dict.iteritems():
            grp[p][vecstim_ns] = h5py.ExternalLink(vecstim_file,"/Populations/%s/%s" % (p, vecstim_ns))


    
