import numpy as np
import pandas as pd
import h5py

def h5_get_group (h, groupname):
    if groupname in list(h.keys()):
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g

def h5_get_dataset (g, dsetname, **kwargs):
    if dsetname in list(g.keys()):
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset

def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize+len(data),)
    dset.resize(newshape)
    dset[dsize:] = data
    return dset

grp_voxeldb = "voxeldb"

def import_voxeldb (inputfile,outputfile,colsep=' ',bufsize=1000000):

    chunksize = 10 ** 6
    count = 0
    for chunk in pd.read_csv(inputfile, chunksize=chunksize, sep=colsep, \
                             names=['X', 'Y', 'Z', 'Longitudinal', 'Transverse', 'Depth', 'Bregma', 'Interaural', 'Type']):
        print('chunk %d read' % count)

        with h5py.File(outputfile, "a", libver="latest") as h5:

            g = h5_get_group (h5, grp_voxeldb)
            g1 = h5_get_group (g, 'Hippocampus')
        

            for col in ['X', 'Y', 'Z', 'Longitudinal', 'Transverse', 'Depth', 'Bregma', 'Interaural']:
                dset = h5_get_dataset(g1, col, dtype=np.float32,
                                      maxshape=(None,), compression=9, shuffle=True)
                dset = h5_concat_dataset(dset, np.asarray(chunk[col]))
            dset = h5_get_dataset(g1, 'Type', dtype=np.int8,
                                  maxshape=(None,), compression=9, shuffle=True)
            dset = h5_concat_dataset(dset, np.asarray(chunk['Type']))

            count = count+1


import_voxeldb("hippocampus-voxeldb.txt",'Hippocampus_voxeldb.h5')


