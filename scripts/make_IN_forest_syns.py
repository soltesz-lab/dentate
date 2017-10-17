import h5py

f = h5py.File("DG_IN_forest_syns_20171013.h5")

grp = f.create_group("Populations")

grp["AAC"] = h5py.ExternalLink("AAC_forest_syns_20171013.h5","/Populations/AAC")
grp["BC"]  = h5py.ExternalLink("BC_forest_syns_20171013.h5","/Populations/BC")
grp["HC"]  = h5py.ExternalLink("HC_forest_syns_20171013.h5","/Populations/HC")
grp["HCC"] = h5py.ExternalLink("HCC_forest_syns_20171013.h5","/Populations/HCC")
grp["IS"]  = h5py.ExternalLink("IS_forest_syns_20171013.h5","/Populations/IS")
grp["MC"]  = h5py.ExternalLink("MC_forest_syns_20171013.h5","/Populations/MC")
grp["MOPP"] = h5py.ExternalLink("MOPP_forest_syns_20171013.h5","/Populations/MOPP")
grp["NGFC"] = h5py.ExternalLink("NGFC_forest_syns_20171013.h5","/Populations/NGFC")

f.close()
