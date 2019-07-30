import os
import h5py
import numpy as np
import dentate
from dentate.utils import Struct, range, str, viewitems, basestring, Iterable, get_module_logger
from neuroh5.io import write_cell_attributes, append_cell_attributes
from neuron import h


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


grp_h5types = 'H5Types'
grp_projections = 'Projections'
grp_populations = 'Populations'

path_population_labels = '/%s/Population labels' % grp_h5types
path_population_range = '/%s/Population range' % grp_h5types

grp_population_projections = 'Population projections'
grp_valid_population_projections = 'Valid population projections'
path_population_projections = '/%s/Population projections' % grp_h5types

# Default I/O configuration
default_io_options = Struct(io_size=-1, chunk_size=1000, value_chunk_size=1000, cache_size=50, write_size=10000)


class CheckpointEvent(object):

    def __init__(self, env, dt_checkpoint=500.0):
        if (int(env.pc.id()) == 0):
            logger.info("*** checkpoint interval is %.2f ms of simulation time" % dt_checkpoint)
        self.env = env
        self.last_checkpoint = 0.
        self.dt_checkpoint = dt_checkpoint
        self.fih_checkpoint = h.FInitializeHandler(1, self.checkpoint)


    def checkpoint(self):
        if (h.t > 0):
            rank = int(self.env.pc.id())
            if rank == 0:
                logger.info("*** Writing spike data up to %.2f ms" % h.t)
            spikeout(self.env, self.env.results_file_path, clear_data=True)
            if self.env.vrecord_fraction > 0.:
                if rank == 0:
                    logger.info("*** Writing intracellular data up to %.2f ms" % h.t)
                recsout(self.env, self.env.results_file_path, t_start=self.last_checkpoint, clear_data=True)
            
        self.last_checkpoint = h.t
        if (h.t + self.dt_checkpoint) < h.tstop:
            h.cvode.event(h.t + self.dt_checkpoint, self.checkpoint)


def h5_get_group(h, groupname):
    if groupname in h:
        g = h[groupname]
    else:
        g = h.create_group(groupname)
    return g


def h5_get_dataset(g, dsetname, **kwargs):
    if dsetname in g:
        dset = g[dsetname]
    else:
        dset = g.create_dataset(dsetname, (0,), **kwargs)
    return dset


def h5_concat_dataset(dset, data):
    dsize = dset.shape[0]
    newshape = (dsize + len(data),)
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
    mapping = {name: idx for name, idx in viewitems(env.Populations)}
    dt_population_labels = h5py.special_dtype(enum=(np.uint16, mapping))

    with h5py.File(output_path, "a") as h5:

        h5[path_population_labels] = dt_population_labels

        dt_populations = np.dtype([("Start", np.uint64), ("Count", np.uint32),
                                   ("Population", h5[path_population_labels].dtype)])
        h5[path_population_range] = dt_populations

        # create an HDF5 compound type for population ranges
        dt = h5[path_population_range].dtype

        g = h5_get_group(h5, grp_h5types)

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
    dataset_path = os.path.join(env.dataset_prefix, env.datasetName)
    data_file_path = os.path.join(dataset_path, env.modelConfig['Cell Data'])
    data_file = h5py.File(data_file_path, 'r')
    results_file = h5py.File(results_filename)
    if 'H5Types' not in results_file:
        data_file.copy('/H5Types', results_file)
    data_file.close()
    results_file.close()


def spikeout(env, output_path, clear_data=False):
    """
    Writes spike time to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :param clear_data: 
    :return:
    """

    t_vec = np.array(env.t_vec, dtype=np.float32)
    id_vec = np.array(env.id_vec, dtype=np.uint32)

    binlst = []
    typelst = sorted(env.celltypes.keys())
    for k in typelst:
        binlst.append(env.celltypes[k]['start'])

    binvect = np.array(binlst)
    sort_idx = np.argsort(binvect, axis=0)
    pop_names = sorted([typelst[i] for i in sort_idx])
    bins = binvect[sort_idx][1:]
    inds = np.digitize(id_vec, bins)

    if env.results_id is None:
        namespace_id = "Spike Events"
    else:
        namespace_id = "Spike Events %s" % str(env.results_id)

    for i, pop_name in enumerate(pop_names):
        spkdict = {}
        sinds = np.where(inds == i)
        if len(sinds) > 0:
            ids = id_vec[sinds]
            ts = t_vec[sinds]
            for j in range(0, len(ids)):
                gid = ids[j]
                t = ts[j]
                if gid in spkdict:
                    spkdict[gid]['t'].append(t)
                else:
                    spkdict[gid] = {'t': [t]}
            for gid in spkdict:
                spkdict[gid]['t'] = np.array(spkdict[gid]['t'], dtype=np.float32)
                if gid in env.spike_onset_delay:
                    spkdict[gid]['t'] -= env.spike_onset_delay[gid]
                
        append_cell_attributes(output_path, pop_name, spkdict, namespace=namespace_id, comm=env.comm, io_size=env.io_size)
        del (spkdict)

    if clear_data:
        env.t_vec.resize(0)
        env.id_vec.resize(0)


def recsout(env, output_path, t_start=0., clear_data=False):
    """
    Writes intracellular voltage traces to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :param clear_data:
    :return:
    """
    t_vec = np.arange(t_start, env.tstop + env.dt, env.dt, dtype=np.float32)

    for pop_name in sorted(env.celltypes.keys()):
        for rec_type, recs in sorted(viewitems(env.recs_dict[pop_name])):
            attr_dict = {}
            for rec in recs:
                gid = rec['gid']
                attr_dict[gid] = {'v': np.array(rec['vec'], copy=clear_data, dtype=np.float32), 't': t_vec}
                if clear_data:
                    rec['vec'].resize(0)
            if env.results_id is None:
                namespace_id = "Intracellular Voltage %s" % rec_type
            else:
                namespace_id = "Intracellular Voltage %s %s" % (rec_type, str(env.results_id))
            append_cell_attributes(output_path, pop_name, attr_dict, namespace=namespace_id, comm=env.comm, io_size=env.io_size)
            

def lfpout(env, output_path):
    """
    Writes local field potential voltage traces to specified HDF5 output file.

    :param env:
    :param output_path:
    :param clear_data:
    :return:
    """

    for lfp in list(env.lfp.values()):

        if env.results_id is None:
            namespace_id = "Local Field Potential %s" % str(lfp.label)
        else:
            namespace_id = "Local Field Potential %s %s" % (str(lfp.label), str(env.results_id))
        import h5py
        output = h5py.File(output_path)

        grp = output.create_group(namespace_id)

        grp['t'] = np.asarray(lfp.t, dtype=np.float32)
        grp['v'] = np.asarray(lfp.meanlfp, dtype=np.float32)

        output.close()


def get_h5py_attr(attrs, key):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always returning them to str type upon read. Values should be converted during write with the companion function
    set_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :return: val with type converted if str or array of str
    """
    if key not in attrs:
        raise KeyError('get_h5py_attr: invalid key: %s' % key)
    val = attrs[key]
    if isinstance(val, basestring):
        val = np.string_(val).astype(str)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], basestring):
            val = np.array(val, dtype='str')
    return val


def set_h5py_attr(attrs, key, val):
    """
    str values are stored as bytes in h5py container attrs dictionaries. This function enables py2/py3 compatibility by
    always converting them to np.string_ upon write. Values should be converted back to str during read with the
    companion function get_h5py_str_attr.
    :param attrs: :class:'h5py._hl.attrs.AttributeManager'
    :param key: str
    :param val: type converted if str or array of str
    """
    if isinstance(val, basestring):
        val = np.string_(val)
    elif isinstance(val, Iterable) and len(val) > 0:
        if isinstance(val[0], basestring):
            val = np.array(val, dtype='S')
    attrs[key] = val
