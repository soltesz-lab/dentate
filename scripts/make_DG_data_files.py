import h5py
import itertools

populations = ["AAC", "BC", "GC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC", "MPP", "LPP"]
IN_populations = ["AAC", "BC", "HC", "HCC", "IS", "MC", "MOPP", "NGFC"]
forest_files = ["AAC_forest_syns_20171013.h5",
                "BC_forest_syns_20171013.h5",
                "DGC_forest_syns_20171024_compressed.h5",
                "HC_forest_syns_20171013.h5",
                "HCC_forest_syns_20171013.h5",
                "IS_forest_syns_20171013.h5",
                "MC_forest_syns_20171013.h5",
                "MOPP_forest_syns_20171013.h5",
                "NGFC_forest_syns_20171013.h5",
                None,
                None]

coords_file = "dentate_Full_Scale_Control_spiketrains_20171024.h5"

vecstim_populations = ["MPP", "LPP"]
spiketrains_file = "dentate_Full_Scale_Control_spiketrains_20171024.h5"

f = h5py.File("DG_cells_20171024.h5")

grp = f.create_group("Populations")
                
for p in populations:
    grp.create_group(p)

for p in populations:
    grp[p]["Coordinates"] = h5py.ExternalLink(coords_file,"/Populations/%s/Coordinates" % p)
    grp[p]["Arc Distance"] = h5py.ExternalLink(coords_file,"/Populations/%s/Arc Distance" % p)

for p in vecstim_populations:
    grp[p]["Vector Stimulus 0"] = h5py.ExternalLink(spiketrains_file,"/Populations/%s/Vector Stimulus 0" % p)

for (p,ff) in itertools.izip (populations,forest_files):
    if ff is not None:
        grp[p]["Trees"] = h5py.ExternalLink(ff,"/Populations/%s/Trees" % p)
        grp[p]["Synapse Attributes"] = h5py.ExternalLink(ff,"/Populations/%s/Synapse Attributes" % p)

f.close()


IN_connectivity_file = "DG_IN_connections_20171025.h5"
GC_connectivity_file = "DG_GC_connections_20171022.h5"

f = h5py.File("DG_connectivity_20171025.h5")

grp = f.create_group("Projections")

for p in IN_populations:
    grp[p] = h5py.ExternalLink(IN_connectivity_file,"/Projections/%s" % p)
    
grp['GC'] = h5py.ExternalLink(GC_connectivity_file,"/Projections/%s" % 'GC')

f.close()
    
