"""
Dentate Gyrus network initialization routines.
"""
__author__ = 'See AUTHORS.md'

from dentate.neuron_utils import *
import dentate.cells as cells
import dentate.synapses as synapses
import dentate.lpt as lpt
import dentate.lfp as lfp
import dentate.simtime as simtime
import h5py
from neuroh5.io import scatter_read_graph, bcast_graph, scatter_read_trees, scatter_read_cell_attributes, \
    write_cell_attributes


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


# Code by Michael Hines from this discussion thread:
# https://www.neuron.yale.edu/phpBB/viewtopic.php?f=31&t=3628
def cx(env):
    """
    Estimates cell complexity. Uses the LoadBalance class.

    :param env: an instance of the `dentate.Env` class.
    """
    rank = int(env.pc.id())
    lb = h.LoadBalance()
    if os.path.isfile("mcomplex.dat"):
        lb.read_mcomplex()
    cxvec = h.Vector(len(env.gidlist))
    for i, gid in enumerate(env.gidlist):
        cxvec.x[i] = lb.cell_complexity(env.pc.gid2cell(gid))
    env.cxvec = cxvec
    return cxvec


def ld_bal(env):
    """
    For given cxvec on each rank, calculates the fractional load balance.

    :param env: an instance of the `dentate.Env` class.
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    cxvec = env.cxvec
    sum_cx = sum(cxvec)
    max_sum_cx = env.pc.allreduce(sum_cx, 2)
    sum_cx = env.pc.allreduce(sum_cx, 1)
    if rank == 0:
        logger.info("*** expected load balance %.2f" % (sum_cx / nhosts / max_sum_cx))


def lpt_bal(env):
    """
    Load-balancing based on the LPT algorithm. 
    Each rank has gidvec, cxvec: gather everything to rank 0, do lpt
    algorithm and write to a balance file.    

    :param env: an instance of the `dentate.Env` class.
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    cxvec = env.cxvec
    gidvec = env.gidlist
    # gather gidvec, cxvec to rank 0
    src = [None] * nhosts
    src[0] = zip(cxvec.to_python(), gidvec)
    dest = env.pc.py_alltoall(src)
    del src

    if rank == 0:
        lb = h.LoadBalance()
        allpairs = sum(dest, [])
        del dest
        parts = lpt.lpt(allpairs, nhosts)
        lpt.statistics(parts)
        part_rank = 0
        with open('parts.%d' % nhosts, 'w') as fp:
            for part in parts:
                for x in part[1]:
                    fp.write('%d %d\n' % (x[1], part_rank))
                part_rank = part_rank + 1


def mkout(env, results_filename):
    """
    Creates simulation results file and adds H5Types group compatible with NeuroH5.

    :param env:
    :param results_filename:
    :return:
    """
    datasetPath   = os.path.join(env.datasetPrefix,env.datasetName)
    dataFilePath  = os.path.join(datasetPath,env.modelConfig['Cell Data'])
    dataFile      = h5py.File(dataFilePath,'r')
    resultsFile   = h5py.File(results_filename,'w')
    dataFile.copy('/H5Types',resultsFile)
    dataFile.close()
    resultsFile.close()


def spikeout(env, output_path, t_vec, id_vec):
    """
    Writes spike time to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :param t_vec:
    :param id_vec:
    :return:
    """
    binlst  = []
    typelst = env.celltypes.keys()
    for k in typelst:
        binlst.append(env.celltypes[k]['start'])

    binvect  = np.array(binlst)
    sort_idx = np.argsort(binvect,axis=0)
    bins     = binvect[sort_idx][1:]
    types    = [ typelst[i] for i in sort_idx ]
    inds     = np.digitize(id_vec, bins)

    if not str(env.resultsId):
        namespace_id = "Spike Events"
    else:
        namespace_id = "Spike Events %s" % str(env.resultsId)

    for i in range(0,len(types)):
        spkdict  = {}
        sinds    = np.where(inds == i)
        if len(sinds) > 0:
            ids      = id_vec[sinds]
            ts       = t_vec[sinds]
            for j in range(0,len(ids)):
                id = ids[j]
                t  = ts[j]
                if spkdict.has_key(id):
                    spkdict[id]['t'].append(t)
                else:
                    spkdict[id]= {'t': [t]}
            for j in spkdict.keys():
                spkdict[j]['t'] = np.array(spkdict[j]['t'], dtype=np.float32)
        pop_name = types[i]
        write_cell_attributes(output_path, pop_name, spkdict, namespace=namespace_id, comm=env.comm)
        del(spkdict)


def vout(env, output_path, t_vec, v_dict):
    """
    Writes intracellular voltage traces to specified NeuroH5 output file.

    :param env:
    :param output_path:
    :param t_vec:
    :param v_dict:
    :return:
    """
    if not str(env.resultsId):
        namespace_id = "Intracellular Voltage"
    else:
        namespace_id = "Intracellular Voltage %s" % str(env.resultsId)

    for pop_name, gid_v_dict in v_dict.iteritems():
        attr_dict  = {gid: {'v': np.array(vs, dtype=np.float32), 't': t_vec}
                      for (gid, vs) in gid_v_dict.iteritems()}
        write_cell_attributes(output_path, pop_name, attr_dict, namespace=namespace_id, comm=env.comm)


def lfpout(env, output_path, lfp):
    """
    Writes local field potential voltage traces to specified HDF5 output file.

    :param env:
    :param output_path:
    :param lfp:
    :return:
    """
    namespace_id = "Local Field Potential %s" % str(lfp.label)
    import h5py
    output = h5py.File(output_path)

    grp = output.create_group(namespace_id)

    grp['t'] = np.asarray(lfp.t, dtype=np.float32)
    grp['v'] = np.asarray(lfp.meanlfp, dtype=np.float32)

    output.close()


def connectcells(env, cleanup=True):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for each postsynaptic cell.


    TODO: cleanup might need to be more granular than binary
    :param env:

    """
    connectivityFilePath = env.connectivityFilePath
    forestFilePath = env.forestFilePath
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info('*** Connectivity file path is %s' % connectivityFilePath)
        logger.info('*** Reading projections: ')

    for (postsyn_name, presyn_names) in env.projection_dict.iteritems():

        synapse_config = env.celltypes[postsyn_name]['synapses']
        if synapse_config.has_key('correct_for_spines'):
            correct_for_spines = synapse_config['correct_for_spines']
        else:
            correct_for_spines = False

        if synapse_config.has_key('unique'):
            unique = synapse_config['unique']
        else:
            unique = False

        if synapse_config.has_key('weights'):
            has_weights = synapse_config['weights']
        else:
            has_weights = False

        if synapse_config.has_key('weights namespace'):
            weights_namespace = synapse_config['weights namespace']
        else:
            weights_namespace = 'Weights'

        if env.celltypes[postsyn_name].has_key('mech_file'):
            mech_file_path = env.configPrefix + '/' + env.celltypes[postsyn_name]['mech_file']
        else:
            mech_file_path = None

        if rank == 0:
                logger.info('*** Reading synapse attributes of population %s' % (postsyn_name))

        if has_weights:
            cell_attr_namespaces = ['Synapse Attributes', weights_namespace]
        else:
            cell_attr_namespaces = ['Synapse Attributes']

        if env.nodeRanks is None:
            cell_attributes_dict = scatter_read_cell_attributes(forestFilePath, postsyn_name,
                                                                namespaces=cell_attr_namespaces, comm=env.comm,
                                                                io_size=env.IOsize)
        else:
            cell_attributes_dict = scatter_read_cell_attributes(forestFilePath, postsyn_name,
                                                                namespaces=cell_attr_namespaces, comm=env.comm,
                                                                node_rank_map=env.nodeRanks,
                                                                io_size=env.IOsize)
        cell_synapses_dict = {k: v for (k, v) in cell_attributes_dict['Synapse Attributes']}
        if cell_attributes_dict.has_key(weights_namespace):
            cell_weights_dict = {k: v for (k, v) in cell_attributes_dict[weights_namespace]}
            first_gid = None
            for gid in cell_weights_dict:
                if first_gid is None:
                    first_gid = gid
                for syn_name in (syn_name for syn_name in cell_weights_dict[gid] if syn_name != 'syn_id'):
                    # TODO: this is here for backwards compatibility; attr_name should be syn_name (e.g. 'AMPA')
                    if syn_name == 'weight':
                        target_syn_name = 'SatAMPA'
                    else:
                        target_syn_name = syn_name
                    syn_attrs.load_syn_weights(gid, target_syn_name, cell_weights_dict[gid]['syn_id'],
                                               cell_weights_dict[gid][syn_name])
                    if rank == 0 and gid == first_gid:
                        logger.info('*** connectcells: population: %s; gid: %i; found %i %s synaptic weights' %
                                    (postsyn_name, gid, len(cell_weights_dict[gid][syn_name]), target_syn_name))
        del cell_attributes_dict

        first_gid = None
        for gid in cell_synapses_dict:
            syn_attrs.load_syn_id_attrs(gid, cell_synapses_dict[gid])
            if mech_file_path is not None:
                if first_gid is None:
                    first_gid = gid
                biophys_cell = cells.BiophysCell(gid=gid, pop_name=postsyn_name, hoc_cell=env.pc.gid2cell(gid), env=env)
                try:
                    cells.init_biophysics(biophys_cell, mech_file_path=mech_file_path, reset_cable=True, from_file=True,
                                    correct_cm=correct_for_spines, correct_g_pas=correct_for_spines, env=env)
                except IndexError:
                    raise IndexError('connectcells: population: %s; gid: %i; could not load biophysics from path: '
                                     '%s' % (postsyn_name, gid, mech_file_path))
                env.biophys_cells[postsyn_name][gid] = biophys_cell
                if rank == 0 and gid == first_gid:
                    logger.info('*** connectcells: population: %s; gid: %i; loaded biophysics from path: %s' %
                                (postsyn_name, gid, mech_file_path))

        for presyn_name in presyn_names:

            env.edge_count[postsyn_name][presyn_name] = 0

            if rank == 0:
                logger.info('*** Connecting %s -> %s' % (presyn_name, postsyn_name))

            if env.nodeRanks is None:
                (graph, a) = scatter_read_graph(connectivityFilePath, comm=env.comm, io_size=env.IOsize,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])
            else:
                (graph, a) = scatter_read_graph(connectivityFilePath, comm=env.comm, io_size=env.IOsize,
                                                node_rank_map=env.nodeRanks,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])

            edge_iter = graph[postsyn_name][presyn_name]

            syn_params_dict = env.connection_config[postsyn_name][presyn_name].mechanisms

            syn_id_attr_index = a[postsyn_name][presyn_name]['Synapses']['syn_id']
            distance_attr_index = a[postsyn_name][presyn_name]['Connections']['distance']

            for (postsyn_gid, edges) in edge_iter:

                postsyn_cell = env.pc.gid2cell(postsyn_gid)
                presyn_gids = edges[0]
                edge_syn_ids = edges[1]['Synapses'][syn_id_attr_index]
                edge_dists = edges[1]['Connections'][distance_attr_index]

                syn_attrs.load_edge_attrs(postsyn_gid, presyn_name, edge_syn_ids, env)

                edge_syn_obj_dict = \
                    synapses.mksyns(postsyn_gid, postsyn_cell, edge_syn_ids, syn_params_dict, env,
                           env.edge_count[postsyn_name][presyn_name],
                           add_synapse=synapses.add_unique_synapse if unique else synapses.add_shared_synapse)

                if rank == 0:
                    if env.edge_count[postsyn_name][presyn_name] == 0:
                        for sec in list(postsyn_cell.all):
                            h.psection(sec=sec)

                for presyn_gid, edge_syn_id, distance in itertools.izip(presyn_gids, edge_syn_ids, edge_dists):
                    for syn_name, syn in edge_syn_obj_dict[edge_syn_id].iteritems():
                        delay = (distance / env.connection_velocity[presyn_name]) + h.dt
                        this_nc = synapses.mknetcon(env.pc, presyn_gid, postsyn_gid, syn, delay)
                        syn_attrs.append_netcon(postsyn_gid, edge_syn_id, syn_name, this_nc)
                        synapses.config_syn(syn_name=syn_name, rules=syn_attrs.syn_param_rules,
                                   mech_names=syn_attrs.syn_mech_names, nc=this_nc, **syn_params_dict[syn_name])

                env.edge_count[postsyn_name][presyn_name] += len(presyn_gids)

        first_gid = None
        # this is a pre-built list to survive change in len during iteration
        local_time = time.time()
        for gid in cell_synapses_dict.keys():
            if first_gid is None:
                first_gid = gid
                this_verbose = True
            else:
                this_verbose = False
            # TODO: update_mech_attrs
            synapses.config_syns_from_mech_attrs(gid, env, postsyn_name, verbose=this_verbose)
            if cleanup:
                syn_attrs.cleanup(gid)
                if gid in env.biophys_cells[postsyn_name]:
                    del env.biophys_cells[postsyn_name][gid]
        logger.info('*** rank: %i: config_syns and cleanup for %s took %i s' %
                    (rank, postsyn_name, time.time() - local_time))


def connectgjs(env):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding half-gap mechanisms on the pre- and post-junction
    cells.
    :param env: :class:'Env'
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath = os.path.join(env.datasetPrefix,env.datasetName)

    gapjunctions = env.gapjunctions
    if env.gapjunctionsFile is None:
        gapjunctionsFilePath = None
    else:
        gapjunctionsFilePath = os.path.join(datasetPath,env.gapjunctionsFile)

    if gapjunctions is not None:

        h('objref gjlist')
        h.gjlist = h.List()
        datasetPath = os.path.join(env.datasetPrefix,env.datasetName)
        (graph, a) = bcast_graph(gapjunctionsFilePath,attributes=True,comm=env.comm)

        ggid = 2e6
        for name in gapjunctions.keys():
            if rank == 0:
                logger.info("*** Creating gap junctions %s" % name)
            prj = graph[name]
            attrmap = a[name]
            weight_attr_idx = attrmap['Weight']+1
            dstbranch_attr_idx = attrmap['Destination Branch']+1
            dstsec_attr_idx = attrmap['Destination Section']+1
            srcbranch_attr_idx = attrmap['Source Branch']+1
            srcsec_attr_idx = attrmap['Source Section']+1
            for destination in sorted(prj.keys()):
                edges = prj[destination]
                sources      = edges[0]
                weights      = edges[weight_attr_idx]
                dstbranches  = edges[dstbranch_attr_idx]
                dstsecs      = edges[dstsec_attr_idx]
                srcbranches  = edges[srcbranch_attr_idx]
                srcsecs      = edges[srcsec_attr_idx]
                for i in range(0,len(sources)):
                    source    = sources[i]
                    srcbranch = srcbranches[i]
                    srcsec    = srcsecs[i]
                    dstbranch = dstbranches[i]
                    dstsec    = dstsecs[i]
                    weight    = weights[i]
                    if env.pc.gid_exists(source):
                        mkgap(env.pc, h.gjlist, source, srcbranch, srcsec, ggid, ggid+1, weight)
                        logger.info('host %d: gap junction: gid = %d branch = %d sec = %d coupling = %g '
                                    'sgid = %d dgid = %d\n' %
                                    (rank, source, srcbranch, srcsec, weight, ggid, ggid+1))
                    if env.pc.gid_exists(destination):
                        mkgap(env.pc, h.gjlist, destination, dstbranch, dstsec, ggid+1, ggid, weight)
                        logger.info('host %d: gap junction: gid = %d branch = %d sec = %d coupling = %g sgid = %d '
                                    'dgid = %d\n' %
                                    (rank, destination, dstbranch, dstsec, weight, ggid+1, ggid))
                    ggid = ggid+2

            del graph[name]


def mkcells(env):
    """
    Instantiates cell templates according to population ranges and NeuroH5 morphology if present.

    :param env:
    :return:
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    v_sample_seed = int(env.modelConfig['Random Seeds']['Intracellular Voltage Sample'])
    ranstream_v_sample = np.random.RandomState()
    ranstream_v_sample.seed(v_sample_seed)

    datasetPath = env.datasetPath
    dataFilePath = env.dataFilePath
    popNames = env.celltypes.keys()
    popNames.sort()

    for popName in popNames:
        if rank == 0:
            logger.info("*** Creating population %s" % popName)
        env.load_cell_template(popName)
        templateClass = getattr(h, env.celltypes[popName]['template'])

        v_sample_set = set([])
        env.v_dict[popName] = {}

        for gid in xrange(env.celltypes[popName]['start'],
                          env.celltypes[popName]['start'] + env.celltypes[popName]['num']):
            if ranstream_v_sample.uniform() <= env.vrecordFraction:
                v_sample_set.add(gid)

        if env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Trees'):
            if rank == 0:
                logger.info("*** Reading trees for population %s" % popName)

            if env.nodeRanks is None:
                (trees, forestSize) = scatter_read_trees(dataFilePath, popName, comm=env.comm, io_size=env.IOsize)
            else:
                (trees, forestSize) = scatter_read_trees(dataFilePath, popName, comm=env.comm, io_size=env.IOsize,
                                                         node_rank_map=env.nodeRanks)
            if rank == 0:
                logger.info("*** Done reading trees for population %s" % popName)

            h.numCells = 0
            i = 0
            for (gid, tree) in trees:
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (popName, gid))

                model_cell = cells.make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, local_id=i,
                                                       dataset_path=datasetPath)
                if rank == 0 and i == 0:
                    for sec in list(model_cell.all):
                        h.psection(sec=sec)
                env.gidlist.append(gid)
                env.cells.append(model_cell)
                env.pc.set_gid2node(gid, rank)
                # Tell the ParallelContext that this cell is a spike source
                # for all other hosts. NetCon is temporary.
                nc = model_cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                # Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                # Record voltages from a subset of cells
                if gid in v_sample_set:
                    v_vec = h.Vector()
                    soma = list(model_cell.soma)[0]
                    v_vec.record(soma(0.5)._ref_v)
                    env.v_dict[popName][gid] = v_vec
                i = i + 1
                h.numCells = h.numCells + 1
            if rank == 0:
                logger.info("*** Created %i cells" % i)

        elif env.cellAttributeInfo.has_key(popName) and env.cellAttributeInfo[popName].has_key('Coordinates'):
            if rank == 0:
                logger.info("*** Reading coordinates for population %s" % popName)

            if env.nodeRanks is None:
                cell_attributes_dict = scatter_read_cell_attributes(dataFilePath, popName,
                                                                    namespaces=['Coordinates'],
                                                                    comm=env.comm, io_size=env.IOsize)
            else:
                cell_attributes_dict = scatter_read_cell_attributes(dataFilePath, popName,
                                                                    namespaces=['Coordinates'],
                                                                    node_rank_map=env.nodeRanks,
                                                                    comm=env.comm, io_size=env.IOsize)
            if rank == 0:
                logger.info("*** Done reading coordinates for population %s" % popName)

            coords = cell_attributes_dict['Coordinates']

            h.numCells = 0
            i = 0
            for (gid, cell_coords_dict) in coords:
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (popName, gid))

                model_cell = cells.make_cell(templateClass, gid=gid, local_id=i, dataset_path=datasetPath)

                cell_x = cell_coords_dict['X Coordinate'][0]
                cell_y = cell_coords_dict['Y Coordinate'][0]
                cell_z = cell_coords_dict['Z Coordinate'][0]
                model_cell.position(cell_x, cell_y, cell_z)

                env.gidlist.append(gid)
                env.cells.append(model_cell)
                env.pc.set_gid2node(gid, rank)
                # Tell the ParallelContext that this cell is a spike source
                # for all other hosts. NetCon is temporary.
                nc = model_cell.connect2target(h.nil)
                env.pc.cell(gid, nc, 1)
                # Record spikes of this cell
                env.pc.spike_record(gid, env.t_vec, env.id_vec)
                i = i + 1
                h.numCells = h.numCells + 1
        h.define_shape()


def mkstim(env):
    """
    Loads spike train data from NeuroH5 file for those populations
    that have 'Vector Stimulus' entry in the cell configuration.

    :param env:
    :return:

    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    datasetPath = env.datasetPath
    inputFilePath = env.dataFilePath

    popNames = env.celltypes.keys()
    popNames.sort()
    for popName in popNames:
        if env.celltypes[popName].has_key('Vector Stimulus'):
            vecstim_namespace = env.celltypes[popName]['Vector Stimulus']

            if env.nodeRanks is None:
                cell_attributes_dict = scatter_read_cell_attributes(inputFilePath, popName,
                                                                    namespaces=[vecstim_namespace],
                                                                    comm=env.comm, io_size=env.IOsize)
            else:
                cell_attributes_dict = scatter_read_cell_attributes(inputFilePath, popName,
                                                                    namespaces=[vecstim_namespace],
                                                                    node_rank_map=env.nodeRanks,
                                                                    comm=env.comm, io_size=env.IOsize)
            cell_vecstim = cell_attributes_dict[vecstim_namespace]
            if rank == 0:
                logger.info("*** Stimulus onset is %g ms" % env.stimulus_onset)
            for (gid, vecstim_dict) in cell_vecstim:
                if len(vecstim_dict['spiketrain']) > 0:
                    logger.info("*** Spike train for gid %i is of length %i (first spike at %g ms)" %
                                (gid, len(vecstim_dict['spiketrain']), vecstim_dict['spiketrain'][0]))
                else:
                    logger.info("*** Spike train for gid %i is of length %i" %
                                (gid, len(vecstim_dict['spiketrain'])))

                vecstim_dict['spiketrain'] += env.stimulus_onset
                cell = env.pc.gid2cell(gid)
                cell.play(h.Vector(vecstim_dict['spiketrain']))


def init(env):
    """
    Initializes the network by calling mkcells, mkstim, connectcells, connectgjs.
    Optionally performs load balancing.

    :param env:
    """
    h.load_file("nrngui.hoc")
    h.load_file("loadbal.hoc")
    h('objref pc, nclist, nc, nil')
    h('strdef datasetPath')
    h('numCells = 0')
    h.nclist = h.List()
    h.datasetPath = env.datasetPath
    #  new ParallelContext object
    h.pc   = h.ParallelContext()
    env.pc = h.pc
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    h.dt = env.dt
    h.tstop = env.tstop
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if rank == 0:
        mkout(env, env.resultsFilePath)
    if rank == 0:
        logger.info("*** Creating cells...")
    h.startsw()

    env.pc.barrier()
    mkcells(env)
    env.mkcellstime = h.stopsw()
    h.startsw()
    env.pc.barrier()
    if rank == 0:
        logger.info("*** Cells created in %g seconds" % env.mkcellstime)
    logger.info("*** Rank %i created %i cells" % (rank, len(env.cells)))
    mkstim(env)
    env.mkstimtime = h.stopsw()
    if rank == 0:
        logger.info("*** Stimuli created in %g seconds" % env.mkstimtime)
    h.startsw()
    env.pc.barrier()
    connectcells(env)
    env.pc.barrier()
    env.connectcellstime = h.stopsw()
    if rank == 0:
        logger.info("*** Connections created in %g seconds" % env.connectcellstime)
    edge_count = int(sum([env.edge_count[dest][source] for dest in env.edge_count for source in env.edge_count[dest]]))
    logger.info("*** Rank %i created %i connections" % (rank, edge_count))
    h.startsw()
    connectgjs(env)
    env.pc.setup_transfer()
    env.pc.set_maxstep(10.0)
    env.connectgjstime = h.stopsw()
    if rank == 0:
        logger.info("*** Gap junctions created in %g seconds" % env.connectgjstime)
    h.startsw()
    for lfp_label,lfp_config_dict in env.lfpConfig.iteritems():
        env.lfp[lfp_label] = \
            lfp.LFP(lfp_label, env.pc, env.celltypes, lfp_config_dict['position'], rho=lfp_config_dict['rho'],
                    dt_lfp=lfp_config_dict['dt'], fdst=lfp_config_dict['fraction'],
                    maxEDist=lfp_config_dict['maxEDist'],
                    seed=int(env.modelConfig['Random Seeds']['Local Field Potential']))
    setup_time           = env.mkcellstime + env.mkstimtime + env.connectcellstime + env.connectgjstime + h.stopsw()
    max_setup_time       = self.pc.allreduce(setup_time, 2) ## maximum value
    env.simtime          = simtime.SimTimeEvent(env.pc, env.max_walltime_hrs, env.results_write_time, max_setup_time)
    h.v_init = env.v_init
    h.stdinit()
    h.finitialize(env.v_init)
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)


def run(env, output=True):
    """
    Runs network simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env:
    :param output: bool

    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if rank == 0:
        logger.info("*** Running simulation")

    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")
    del env.cells
    env.pc.barrier()
    if rank == 0:
        logger.info("*** Writing spike data")
    if output:
        spikeout(env, env.resultsFilePath, np.array(env.t_vec, dtype=np.float32), np.array(env.id_vec, dtype=np.uint32))
        if env.vrecordFraction > 0.:
          if rank == 0:
            logger.info("*** Writing intracellular trace data")
          t_vec = np.arange(0, h.tstop+h.dt, h.dt, dtype=np.float32)
          vout(env, env.resultsFilePath, t_vec, env.v_dict)
        env.pc.barrier()
        if rank == 0:
            logger.info("*** Writing local field potential data")
            for lfp in env.lfp.itervalues():
                lfpout(env, env.resultsFilePath, lfp)

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    avgcomp  = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    if rank == 0:
        logger.info("Execution time summary for host %i:" % rank)
        logger.info("  created cells in %g seconds" % env.mkcellstime)
        logger.info("  connected cells in %g seconds" % env.connectcellstime)
        logger.info("  created gap junctions in %g seconds" % env.connectgjstime)
        logger.info("  ran simulation in %g seconds" % comptime)
        if maxcw > 0:
            logger.info("  load balance = %g" % (avgcomp/maxcw))

    env.pc.runworker()
    env.pc.done()
    h.quit()
