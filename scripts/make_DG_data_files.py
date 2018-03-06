import h5py
import itertools

DG_populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP"]
DG_IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
DG_EXT_populations = ["MPP", "LPP"]

DG_GC_coordinate_file  = "DGC_forest_reindex_compressed_20180224.h5"
DG_IN_coordinate_file  = "dentate_generated_coords_20180305.h5"
DG_EXT_coordinate_file = "dentate_generated_coords_20180305.h5"

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
    

DG_cells_file = "DG_cells_20180228.h5"

f = h5py.File(DG_cells_file)

grp = f.create_group("Populations")
                
for p in DG_populations:
    grp.create_group(p)

for p in DG_populations:
    coords_file = coordinate_files[p]
    coords_ns   = coordinate_namespaces[p]
    grp[p]["Coordinates"] = h5py.ExternalLink(coords_file,"/Populations/%s/%s" % (p, coords_ns))

f.close()

# DG_IN_forest_file = "DG_IN_forest_syns_20171102.h5"

# forest_files = {
#     'AAC': DG_IN_forest_file,
#     'BC': DG_IN_forest_file,
#     'GC': "DGC_forest_syns_20171024_compressed.h5",
#     'HC': DG_IN_forest_file,
#     'HCC': DG_IN_forest_file,
#     'IS': DG_IN_forest_file,
#     'MC': DG_IN_forest_file,
#     'MOPP': DG_IN_forest_file,
#     'NGFC': DG_IN_forest_file 
# }

# coords_file = "dentate_Full_Scale_Control_coords_PP_spiketrains_20171107.h5"

# vecstim_populations = ["MPP", "LPP"]

# spiketrains_dict = {
#     'Vector Stimulus 0': "dentate_Full_Scale_Control_coords_PP_spiketrains_20171107.h5"
# }

# DG_cells_file = "DG_cells_20171116.h5"

# f = h5py.File(DG_cells_file)

# grp = f.create_group("Populations")
                
# for p in populations:
#     grp.create_group(p)

# for p in populations:
#     grp[p]["Coordinates"] = h5py.ExternalLink(coords_file,"/Populations/%s/Coordinates" % p)
#     grp[p]["Arc Distance"] = h5py.ExternalLink(coords_file,"/Populations/%s/Arc Distance" % p)

# for (p,ff) in itertools.izip (populations,forest_files):
#     if ff is not None:
#         grp[p]["Trees"] = h5py.ExternalLink(ff,"/Populations/%s/Trees" % p)
#         grp[p]["Synapse Attributes"] = h5py.ExternalLink(ff,"/Populations/%s/Synapse Attributes" % p)

# for p in vecstim_populations:
#     for (vecstim_ns, spiketrains_file) in spiketrains_dict.iteritems():
#         grp[p][vecstim_ns] = h5py.ExternalLink(spiketrains_file,"/Populations/%s/%s" % (p, vecstim_ns))

# f.close()


# IN_connectivity_file = "DG_IN_connections_20171025.h5"
# GC_connectivity_file = "DG_GC_connections_20171022.h5"

# f = h5py.File("DG_connectivity_20171025.h5")

# grp = f.create_group("Projections")

# for p in IN_populations:
#     grp[p] = h5py.ExternalLink(IN_connectivity_file,"/Projections/%s" % p)
    
# grp['GC'] = h5py.ExternalLink(GC_connectivity_file,"/Projections/%s" % 'GC')

# f.close()
    
