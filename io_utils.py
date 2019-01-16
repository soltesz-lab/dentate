import sys, os, itertools
from collections import defaultdict
import h5py
import numpy as np
from dentate import utils
from utils import viewitems
from neuroh5.io import write_cell_attributes

grp_h5types      = 'H5Types'
grp_projections  = 'Projections'
grp_populations  = 'Populations'

path_population_labels = '/%s/Population labels' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types

grp_population_projections = 'Population projections'
grp_valid_population_projections = 'Valid population projections'
path_population_projections = '/%s/Population projections' % grp_h5types


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


def make_h5types(env, output_path, gap_junctions=False):

    populations = []
    for pop_name, pop_idx in viewitems(env.Populations):
        layer_counts = env.geometry['Cell Layer Counts'][pop_name]
        pop_count = 0
        for layer_name, layer_count in viewitems(layer_counts):
            pop_count += layer_count
        populations.append((pop_name, pop_idx, pop_count))
    populations.sort(key=lambda x: x[1])

    projections = []
    if gap_junctions:
        for (post, pre), connection_dict in viewitems(env.gapjunctions):
            projections.append((env.Populations[pre], env.Populations[post]))
    else:
        for post, connection_dict in viewitems(env.connection_config):
            for pre, _ in viewitems(connection_dict):
                projections.append((env.Populations[pre], env.Populations[post]))
    
    # create an HDF5 enumerated type for the population label
    mapping = { name: idx for name, idx in viewitems(env.Populations) }
    dt_population_labels = h5py.special_dtype(enum=(np.uint16, mapping))

    with h5py.File(output_path, "a") as h5:


        h5[path_population_labels] = dt_population_labels

        dt_populations = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                                   ("Population", h5[path_population_labels].dtype)])
        h5[path_population_range]  = dt_populations
        
        # create an HDF5 compound type for population ranges
        dt = h5[path_population_range].dtype

        g = h5_get_group (h5, grp_h5types)

        dset = h5_get_dataset(g, grp_populations, maxshape=(len(populations),), dtype=dt)
        dset.resize((len(populations),))
        a = np.zeros(len(populations), dtype=dt)

        start = 0
        for name, idx, count in populations:
            a[idx]["Start"] = start
            a[idx]["Count"] = count
            a[idx]["Population"] = idx
            start += count

        dset[:] = a

        dt_projections = np.dtype([("Source", h5[path_population_labels].dtype),
                                    ("Destination", h5[path_population_labels].dtype)])

        h5[path_population_projections] = dt_projections

        dt = h5[path_population_projections]
        dset = h5_get_dataset(g, grp_valid_population_projections,
                              maxshape=(len(projections),), dtype=dt)
        dset.resize((len(projections),))
        a = np.zeros(len(projections), dtype=dt)
        idx = 0
        for i, prj in enumerate(projections):
            src, dst = prj
            a[i]["Source"] = int(src)
            a[i]["Destination"] = int(dst)

        dset[:] = a


def mkout(env, results_filename):
    """
    Creates simulation results file and adds H5Types group compatible with NeuroH5.

    :param env:
    :param results_filename:
    :return:
    """
    dataset_path   = os.path.join(env.dataset_prefix, env.datasetName)
    data_file_path  = os.path.join(dataset_path,env.modelConfig['Cell Data'])
    data_file      = h5py.File(data_file_path,'r')
    results_file   = h5py.File(results_filename,'w')
    data_file.copy('/H5Types',results_file)
    data_file.close()
    results_file.close()


def spikeout(env, output_path):
    """
    Writes spike time to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :return:
    """

    t_vec = np.array(env.t_vec, dtype=np.float32)
    id_vec = np.array(env.id_vec, dtype=np.uint32)

    binlst  = []
    typelst = list(env.celltypes.keys())
    for k in typelst:
        binlst.append(env.celltypes[k]['start'])

    binvect  = np.array(binlst)
    sort_idx = np.argsort(binvect,axis=0)
    bins     = binvect[sort_idx][1:]
    types    = [ typelst[i] for i in sort_idx ]
    inds     = np.digitize(id_vec, bins)

    if env.results_id is None:
        namespace_id = "Spike Events"
    else:
        namespace_id = "Spike Events %s" % str(env.results_id)

    for i in range(0,len(types)):
        spkdict  = {}
        sinds    = np.where(inds == i)
        if len(sinds) > 0:
            ids      = id_vec[sinds]
            ts       = t_vec[sinds]
            for j in range(0,len(ids)):
                id = ids[j]
                t  = ts[j]
                if id in spkdict:
                    spkdict[id]['t'].append(t)
                else:
                    spkdict[id]= {'t': [t]}
            for j in list(spkdict.keys()):
                spkdict[j]['t'] = np.array(spkdict[j]['t'], dtype=np.float32)
        pop_name = types[i]
        write_cell_attributes(output_path, pop_name, spkdict, namespace=namespace_id, comm=env.comm)
        del(spkdict)


def recsout(env, output_path):
    """
    Writes intracellular voltage traces to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :param recs:
    :return:
    """
    t_vec = np.arange(0, env.tstop+env.dt, env.dt, dtype=np.float32)
    
    for pop_name in sorted(list(env.celltypes.keys())):
        for rec_type, recs in viewitems(env.recs_dict[pop_name]):
            attr_dict = {}
            for rec in recs:
                gid = rec['gid']
                attr_dict[gid] = {'v': np.array(rec['vec'], dtype=np.float32), 't': t_vec} 
            namespace_id = "Intracellular Voltage %s" % rec_type
            write_cell_attributes(output_path, pop_name, attr_dict, \
                                  namespace=namespace_id, comm=env.comm)


def lfpout(env, output_path):
    """
    Writes local field potential voltage traces to specified HDF5 output file.

    :param env:
    :param output_path:
    :param lfp:
    :return:
    """

    for lfp in env.lfp.values():

        namespace_id = "Local Field Potential %s" % str(lfp.label)
        import h5py
        output = h5py.File(output_path)
        
        grp = output.create_group(namespace_id)
        
        grp['t'] = np.asarray(lfp.t, dtype=np.float32)
        grp['v'] = np.asarray(lfp.meanlfp, dtype=np.float32)
        
        output.close()
