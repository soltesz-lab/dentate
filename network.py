"""
Dentate Gyrus network initialization routines.
"""
__author__ = 'See AUTHORS.md'

import itertools, gc
import dentate
from dentate.neuron_utils import *
from dentate.utils import viewitems, zip_longest, compose_iter, profile_memory
from dentate import cells, synapses, lpt, lfp, simtime, io_utils
import h5py
from neuroh5.io import scatter_read_graph, bcast_graph, \
     scatter_read_trees, scatter_read_cell_attributes, \
     write_cell_attributes, read_cell_attribute_selection, \
     read_tree_selection, read_graph_selection
from neuron import h

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)



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
    gidvec = list(env.gidset)
    # gather gidvec, cxvec to rank 0
    src = [None] * nhosts
    src[0] = list(zip(cxvec.to_python(), gidvec))
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


def register_cell(env, pop_name, gid, cell):
    """
    Registers a cell in a network environment.

    :param env: an instance of the `dentate.Env` class
    :param pop_name: population name
    :param gid: gid
    :param cell: cell instance
    """
    rank = env.comm.rank
    env.gidset.add(gid)
    env.cells.append(cell)
    env.pc.set_gid2node(gid, rank)
    # Tell the ParallelContext that this cell is a spike source
    # for all other hosts. NetCon is temporary.
    nc = cell.connect2target(h.nil)
    nc.delay = env.dt
    env.pc.cell(gid, nc, 1)
    # Record spikes of this cell
    env.pc.spike_record(gid, env.t_vec, env.id_vec)


def connect_cells(env, cleanup=True):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for each postsynaptic cell.

    :param env: an instance of the `dentate.Env` class
    :param cleanup: if true, free synapse attribute dictionaries after synapses and netcons are instantiated.
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info('*** Connectivity file path is %s' % connectivity_file_path)
        logger.info('*** Reading projections: ')

    for (postsyn_name, presyn_names) in viewitems(env.projection_dict):

        if rank == 0:
            logger.info('*** Reading projections of population %s' % postsyn_name)

        synapse_config = env.celltypes[postsyn_name]['synapses']
        if 'correct_for_spines' in synapse_config:
            correct_for_spines = synapse_config['correct_for_spines']
        else:
            correct_for_spines = False

        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

        if 'weights' in synapse_config:
            has_weights = synapse_config['weights']
        else:
            has_weights = False

        if 'weights namespace' in synapse_config:
            weights_namespace = synapse_config['weights namespace']
        else:
            weights_namespace = 'Weights'

        if 'mech_file_path' in env.celltypes[postsyn_name]:
            mech_file_path = env.celltypes[postsyn_name]['mech_file_path']
        else:
            mech_file_path = None

        if rank == 0:
            logger.info('*** Mechanism file for population %s is %s' % (postsyn_name, str(mech_file_path)))

        if rank == 0:
            logger.info('*** Reading synapse attributes of population %s' % (postsyn_name))

        if has_weights:
            cell_attr_namespaces = ['Synapse Attributes', weights_namespace]
        else:
            cell_attr_namespaces = ['Synapse Attributes']

        if env.node_ranks is None:
            cell_attributes_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                namespaces=cell_attr_namespaces, comm=env.comm,
                                                                io_size=env.io_size)
        else:
            cell_attributes_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                namespaces=cell_attr_namespaces, comm=env.comm,
                                                                node_rank_map=env.node_ranks,
                                                                io_size=env.io_size)
        syn_attrs.init_syn_id_attrs_from_iter(cell_attributes_dict['Synapse Attributes'])
        del cell_attributes_dict['Synapse Attributes']
        
        if weights_namespace in cell_attributes_dict:
            syn_weights_iter = cell_attributes_dict[weights_namespace]
            first_gid = None
            for gid, cell_weights_dict in syn_weights_iter:
                if first_gid is None:
                    first_gid = gid
                weights_syn_ids = cell_weights_dict['syn_id']
                for syn_name in (syn_name for syn_name in cell_weights_dict if syn_name != 'syn_id'):
                    if syn_name not in syn_attrs.syn_mech_names:
                        logger.info('*** connect_cells: population: %s; gid: %i; syn_name: %s '
                                    'not found in network configuration' % 
                                    (postsyn_name, gid, syn_name))
                        raise Exception
                    weights_values  = cell_weights_dict[syn_name]
                    syn_attrs.add_mech_attrs_from_iter(gid, syn_name,
                                                       zip_longest(weights_syn_ids,
                                                                   itertools.imap(lambda x: {'weight': x},
                                                                                  weights_values)))
                    if rank == 0 and gid == first_gid:
                        logger.info('*** connect_cells: population: %s; gid: %i; found %i %s synaptic weights' %
                                    (postsyn_name, gid, len(cell_weights_dict[syn_name]), syn_name))

            del cell_attributes_dict[weights_namespace]

        first_gid = None
        for gid in syn_attrs.gids():
            if mech_file_path is not None:
                if first_gid is None:
                    first_gid = gid
                hoc_cell = env.pc.gid2cell(gid)
                biophys_cell = cells.BiophysCell(gid=gid, pop_name=postsyn_name, hoc_cell=hoc_cell, env=env)
                try:
                    cells.init_biophysics(biophys_cell, mech_file_path=mech_file_path, reset_cable=True,
                                          from_file=True, correct_cm=correct_for_spines,
                                          correct_g_pas=correct_for_spines, env=env,
                                          verbose=((rank == 0) and (first_gid==gid)))
                except IndexError:
                    raise IndexError('*** connect_cells: population: %s; gid: %i; could not load biophysics from path: '
                                     '%s' % (postsyn_name, gid, mech_file_path))
                env.biophys_cells[postsyn_name][gid] = biophys_cell
                if rank == 0 and gid == first_gid:
                    logger.info('*** connect_cells: population: %s; gid: %i; loaded biophysics from path: %s' %
                                (postsyn_name, gid, mech_file_path))

        env.edge_count[postsyn_name] = 0
        for presyn_name in presyn_names:


            if rank == 0:
                logger.info('*** Connecting %s -> %s' % (presyn_name, postsyn_name))

            logger.info('Rank %i: Reading projection %s -> %s' % (rank, presyn_name, postsyn_name))
            if env.node_ranks is None:
                (graph, a) = scatter_read_graph(connectivity_file_path, comm=env.comm, io_size=env.io_size,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])
            else:
                (graph, a) = scatter_read_graph(connectivity_file_path, comm=env.comm, io_size=env.io_size,
                                                node_rank_map=env.node_ranks,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])
            logger.info('Rank %i: Read projection %s -> %s' % (rank, presyn_name, postsyn_name))
            edge_iter = graph[postsyn_name][presyn_name]

            last_time = time.time()
            syn_attrs.init_edge_attrs_from_iter(postsyn_name, presyn_name, a, edge_iter)
            logger.info('Rank %i: took %f s to initialize edge attributes for projection %s -> %s' % \
                         (rank, time.time() - last_time, presyn_name, postsyn_name))
            del graph[postsyn_name][presyn_name]

        first_gid = None
        pop_last_time = time.time()

        gids = syn_attrs.gids()
        comm0 = env.comm.Split(2 if len(gids) > 0 else 0, 0)
        
        for gid in gids:

            if first_gid is None:
                first_gid = gid

            if gid in env.biophys_cells[postsyn_name]:
                biophys_cell = env.biophys_cells[postsyn_name][gid]
                synapses.init_syn_mech_attrs(biophys_cell, env)

            postsyn_cell = env.pc.gid2cell(gid)

            if rank == 0 and gid == first_gid:
                logger.info('Rank %i: configuring synapses for gid %i' % (rank, gid))
            
            last_time = time.time()
            syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(
                env, gid, postsyn_name, cell=postsyn_cell, unique=unique, insert=True, insert_netcons=True)

            if rank == 0 and gid == first_gid:
                logger.info('Rank %i: took %f s to configure %i synapses, %i synaptic mechanisms, %i network '
                            'connections for gid %d' % \
                            (rank, time.time() - last_time, syn_count, mech_count, nc_count, gid))
                hoc_cell = env.pc.gid2cell(gid)
                for sec in list(hoc_cell.all):
                    h.psection(sec=sec)

            if gid == first_gid:
                synapses.sample_syn_mech_attrs(env, postsyn_name, [gid], comm=comm0)
                """
                if rank == 0 and 'AMPA' in syn_attrs.syn_mech_names and gid in env.biophys_cells[postsyn_name]:
                    biophys_cell = env.biophys_cells[postsyn_name][gid]
                    from dentate.plot import plot_synaptic_attribute_distribution
                    results_file = 'example_syn_attrs.h5'
                    overwrite = True
                    syn_mech_name = syn_attrs.syn_mech_names['AMPA']
                    if 'g_unit' in syn_attrs.syn_param_rules[syn_mech_name]['netcon_params']:
                        param_label = 'AMPA: g_unit'
                        plot_synaptic_attribute_distribution(
                            biophys_cell, env, 'AMPA', 'g_unit', from_mech_attrs=True, from_target_attrs=True,
                            param_label=param_label, export=results_file, description='gid %i; after' % gid,
                            show=False, overwrite=overwrite, data_dir=env.results_path)
                    overwrite = False
                    param_label = 'AMPA: weight'
                    plot_synaptic_attribute_distribution(
                        biophys_cell, env, 'AMPA', 'weight', from_mech_attrs=True, from_target_attrs=True,
                        param_label=param_label, export=results_file, description='gid %i; after' % gid,
                        show=False, overwrite=overwrite, data_dir=env.results_path)
                """

            env.edge_count[postsyn_name] += syn_count

            if cleanup:
                syn_attrs.del_syn_id_attr_dict(gid)
                if gid in env.biophys_cells[postsyn_name]:
                    del env.biophys_cells[postsyn_name][gid]

        comm0.Free()
        gc.collect()

        if rank == 0:
            logger.info('Rank %i: took %f s to configure synapses for population %s' %
                        (rank, time.time() - pop_last_time, postsyn_name))

def find_gid_pop(celltypes, gid):
    """
    Given a celltypes structure and a gid, find the population to which the gid belongs.
    """
    for pop_name in celltypes.keys():
        start = celltypes[pop_name]['start']
        num = celltypes[pop_name]['num']
        if (start <= gid) and (gid < (start + num)):
           return pop_name

    return None

def connect_cell_selection(env, cleanup=True):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for the selected postsynaptic cells.

    :param env: an instance of the `dentate.Env` class
    :param cleanup: if true, free synapse attribute dictionaries after synapses and netcons are instantiated.
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info('*** Connectivity file path is %s' % connectivity_file_path)
        logger.info('*** Reading projections: ')

    selection_pop_names = env.cell_selection.keys()
    
    vecstim_sources = { pop_name: set([]) for pop_name in env.celltypes.keys() }
    
    for (postsyn_name, presyn_names) in viewitems(env.projection_dict):

        if rank == 0:
            logger.info('*** Postsynaptic population: %s' % postsyn_name)

        if postsyn_name not in selection_pop_names:
            continue

        vecstim_sources[postsyn_name] = set([])
        
        gid_range = [ gid for gid in env.cell_selection[postsyn_name] if gid % nhosts == rank]

        synapse_config = env.celltypes[postsyn_name]['synapses']
        if 'correct_for_spines' in synapse_config:
            correct_for_spines = synapse_config['correct_for_spines']
        else:
            correct_for_spines = False

        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

        if 'weights' in synapse_config:
            has_weights = synapse_config['weights']
        else:
            has_weights = False

        if 'weights namespace' in synapse_config:
            weights_namespace = synapse_config['weights namespace']
        else:
            weights_namespace = 'Weights'

        if 'mech file_path' in env.celltypes[postsyn_name]:
            mech_file_path = env.celltypes[postsyn_name]['mech_file_path']
        else:
            mech_file_path = None

        if rank == 0:
                logger.info('*** Reading synapse attributes of population %s' % (postsyn_name))

        syn_attributes_iter = read_cell_attribute_selection(forest_file_path, postsyn_name, selection=gid_range,
                                                            namespace='Synapse Attributes', comm=env.comm)
        syn_attrs.init_syn_id_attrs_from_iter(syn_attributes_iter)
        del(syn_attributes_iter)
        
        if has_weights:
            weight_attributes_iter = read_cell_attribute_selection(forest_file_path, postsyn_name, selection=gid_range,
                                                                    namespace=weights_namespace, comm=env.comm)
        else:
            weight_attributes_iter = None

        if weight_attributes_iter is not None:
            first_gid = None
            for gid, cell_weights_dict in weight_attributes_iter:
                if first_gid is None:
                    first_gid = gid
                weights_syn_ids = cell_weights_dict['syn_id']
                for syn_name in (syn_name for syn_name in cell_weights_dict if syn_name != 'syn_id'):
                    weights_values  = cell_weights_dict[syn_name]
                    syn_attrs.add_mech_attrs_from_iter(gid, syn_name, \
                                                       zip_longest(weights_syn_ids, \
                                                                   itertools.imap(lambda x: { 'weight' : x }, \
                                                                                  weights_values)))
                    if rank == 0 and gid == first_gid:
                        logger.info('*** connect_cells: population: %s; gid: %i; found %i %s synaptic weights' %
                                    (postsyn_name, gid, len(cell_weights_dict[syn_name]), syn_name))
            del weight_attributes_iter

        first_gid = None
        for gid in syn_attrs.gids():
            if mech_file_path is not None:
                if first_gid is None:
                    first_gid = gid
                biophys_cell = cells.BiophysCell(gid=gid, pop_name=postsyn_name, hoc_cell=env.pc.gid2cell(gid), env=env)
                try:
                    cells.init_biophysics(biophys_cell, \
                                          mech_file_path=mech_file_path, \
                                          reset_cable=True, \
                                          from_file=True, \
                                          correct_cm=correct_for_spines, \
                                          correct_g_pas=correct_for_spines, \
                                          env=env, verbose=((rank == 0) and (first_gid==gid)))
                except IndexError:
                    raise IndexError('connect_cells: population: %s; gid: %i; could not load biophysics from path: '
                                     '%s' % (postsyn_name, gid, mech_file_path))
                env.biophys_cells[postsyn_name][gid] = biophys_cell
                if rank == 0 and gid == first_gid:
                    logger.info('*** connect_cells: population: %s; gid: %i; loaded biophysics from path: %s' %
                                (postsyn_name, gid, mech_file_path))

        (graph, a) = read_graph_selection(connectivity_file_path, selection=gid_range, \
                                          comm=env.comm, namespaces=['Synapses', 'Connections'])
        env.edge_count[postsyn_name] = 0
        for presyn_name in presyn_names:

            if rank == 0:
                logger.info('*** Connecting %s -> %s' % (presyn_name, postsyn_name))
            
            edge_iters = itertools.tee(graph[postsyn_name][presyn_name])

            syn_attrs.init_edge_attrs_from_iter(postsyn_name, presyn_name, a, \
                                                compose_iter(lambda edgeset: vecstim_sources[presyn_name].update(edgeset[1][0]), \
                                                             edge_iters))
            del graph[postsyn_name][presyn_name]

    for pop_name, stim_gid_range in viewitems(vecstim_sources):

        if pop_name in env.cell_selection:
            local_stim_gid_range = stim_gid_range.difference(set(env.cell_selection[pop_name]))
        else:
            local_stim_gid_range = stim_gid_range
        stim_gid_ranges = env.comm.allgather(local_stim_gid_range)
        for gid_range in stim_gid_ranges:
            for gid in gid_range:
                if (gid % nhosts == rank) and not env.pc.gid_exists(gid):
                    stim_cell = h.VecStimCell(gid)
                    register_cell(env, pop_name, gid, stim_cell)

    for gid in syn_attrs.gids():

        cell = env.pc.gid2cell(gid)
        pop_name = find_gid_pop(env.celltypes, gid)
        syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(env, gid, pop_name, \
                                                                        cell=cell, unique=unique, \
                                                                        insert=True, insert_netcons=True)
        env.edge_count[pop_name] += syn_count
        if cleanup:
            syn_attrs.del_syn_id_attr_dict(gid)
            if gid in env.biophys_cells[pop_name]:
                del env.biophys_cells[pop_name][gid]


    return vecstim_sources


def connect_gjs(env):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    half-gap mechanisms on the pre- and post-junction cells.

    :param env: an instance of the `dentate.Env` class

    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = os.path.join(env.dataset_prefix, env.datasetName)

    gapjunctions = env.gapjunctions
    gapjunctions_file_path = env.gapjunctions_file_path 

    num_gj = 0
    num_gj_intra = 0
    num_gj_inter = 0
    if gapjunctions_file_path is not None:

        (graph, a) = bcast_graph(gapjunctions_file_path,\
                                 namespaces=['Coupling strength','Location'],\
                                 comm=env.comm)

        ggid = 2e6
        for name in list(gapjunctions.keys()):
            if rank == 0:
                logger.info("*** Creating gap junctions %s" % str(name))
            prj = graph[name[0]][name[1]]
            attrmap = a[(name[1],name[0])]
            cc_src_idx = attrmap['Coupling strength']['Source']
            cc_dst_idx = attrmap['Coupling strength']['Destination']
            dstsec_idx = attrmap['Location']['Destination section']
            dstpos_idx = attrmap['Location']['Destination position']
            srcsec_idx = attrmap['Location']['Source section']
            srcpos_idx = attrmap['Location']['Source position']

            for src in sorted(prj.keys()):
                edges        = prj[src]
                destinations = edges[0]
                cc_dict      = edges[1]['Coupling strength']
                loc_dict     = edges[1]['Location']
                srcweights   = cc_dict[cc_src_idx]
                dstweights   = cc_dict[cc_dst_idx]
                dstposs      = loc_dict[dstpos_idx]
                dstsecs      = loc_dict[dstsec_idx]
                srcposs      = loc_dict[srcpos_idx]
                srcsecs      = loc_dict[srcsec_idx]
                for i in range(0,len(destinations)):
                    dst       = destinations[i]
                    srcpos    = srcposs[i]
                    srcsec    = srcsecs[i]
                    dstpos    = dstposs[i]
                    dstsec    = dstsecs[i]
                    wgt       = srcweights[i]*0.001
                    if env.pc.gid_exists(src):
                        if rank == 0:
                            logger.info('host %d: gap junction: gid = %d sec = %d coupling = %g '
                                        'sgid = %d dgid = %d\n' %
                                        (rank, src, srcsec, wgt, ggid, ggid+1))
                        cell = env.pc.gid2cell(src)
                        gj = mkgap(env, cell, src, srcpos, srcsec, ggid, ggid+1, wgt)
                    if env.pc.gid_exists(dst):
                        if rank == 0:
                            logger.info('host %d: gap junction: gid = %d sec = %d coupling = %g '
                                        'sgid = %d dgid = %d\n' %
                                        (rank, dst, dstsec, wgt, ggid+1, ggid))
                        cell = env.pc.gid2cell(dst)
                        gj = mkgap(env, cell, dst, dstpos, dstsec, ggid+1, ggid, wgt)
                    ggid = ggid+2
                    if env.pc.gid_exists(src) or env.pc.gid_exists(dst):
                        num_gj += 1
                        if env.pc.gid_exists(src) and env.pc.gid_exists(dst):
                            num_gj_intra += 1
                        else:
                            num_gj_inter += 1
                        

            del graph[name[0]][name[1]]
        
        logger.info('*** host %d: created total %i gap junctions: %i intraprocessor %i interprocessor' %
                    (rank, num_gj, num_gj_intra, num_gj_inter))


def make_cells(env):
    """
    Instantiates cell templates according to population ranges and NeuroH5 morphology if present.

    :param env: an instance of the `dentate.Env` class
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    v_sample_seed = int(env.modelConfig['Random Seeds']['Intracellular Voltage Sample'])
    ranstream_v_sample = np.random.RandomState()
    ranstream_v_sample.seed(v_sample_seed)

    dataset_path = env.dataset_path
    data_file_path = env.data_file_path
    pop_names = sorted(env.celltypes.keys())

    for pop_name in pop_names:
        if rank == 0:
            logger.info("*** Creating population %s" % pop_name)
        env.load_cell_template(pop_name)

        v_sample_set = set([])
        env.v_sample_dict[pop_name] = v_sample_set
        
        for gid in range(env.celltypes[pop_name]['start'],
                         env.celltypes[pop_name]['start'] + env.celltypes[pop_name]['num']):
            if ranstream_v_sample.uniform() <= env.vrecord_fraction:
                v_sample_set.add(gid)

        num_cells = 0
        if (pop_name in env.cellAttributeInfo) and ('Trees' in env.cellAttributeInfo[pop_name]):
            if rank == 0:
                logger.info("*** Reading trees for population %s" % pop_name)

            if env.node_ranks is None:
                (trees, forestSize) = scatter_read_trees(data_file_path, pop_name, comm=env.comm, \
                                                         io_size=env.io_size)
            else:
                (trees, forestSize) = scatter_read_trees(data_file_path, pop_name, comm=env.comm, \
                                                         io_size=env.io_size, node_rank_map=env.node_ranks)
            if rank == 0:
                logger.info("*** Done reading trees for population %s" % pop_name)

            first_gid = None
            for i, (gid, tree) in enumerate(trees):
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))
                    
                if first_gid is None:
                    first_gid = gid

                model_cell = cells.make_hoc_cell(env, pop_name, gid, neurotree_dict=tree)
                if rank == 0 and first_gid == gid:
                    for sec in list(model_cell.all):
                        h.psection(sec=sec)
                register_cell(env, pop_name, gid, model_cell)
                # Record voltages from a subset of cells
                if model_cell.is_art() == 0:
                    if gid in env.v_sample_dict[pop_name]: 
                        rec = make_rec(gid, pop_name, gid, model_cell, \
                                       sec=list(model_cell.soma)[0], \
                                       dt=env.dt, loc=0.5, param='v', \
                                       description='Soma')
                        env.recs_dict[pop_name]['Soma'].append(rec)

                num_cells += 1
            del trees

        elif (pop_name in env.cellAttributeInfo) and ('Coordinates' in env.cellAttributeInfo[pop_name]):
            if rank == 0:
                logger.info("*** Reading coordinates for population %s" % pop_name)

            if env.node_ranks is None:
                cell_attributes_dict = scatter_read_cell_attributes(data_file_path, pop_name,
                                                                    namespaces=['Coordinates'],
                                                                    comm=env.comm, io_size=env.io_size)
            else:
                cell_attributes_dict = scatter_read_cell_attributes(data_file_path, pop_name,
                                                                    namespaces=['Coordinates'],
                                                                    node_rank_map=env.node_ranks,
                                                                    comm=env.comm, io_size=env.io_size)
            if rank == 0:
                logger.info("*** Done reading coordinates for population %s" % pop_name)

            coords = cell_attributes_dict['Coordinates']

            for i, (gid, cell_coords_dict) in enumerate(coords):
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))

                model_cell = cells.make_hoc_cell(env, pop_name, gid)

                cell_x = cell_coords_dict['X Coordinate'][0]
                cell_y = cell_coords_dict['Y Coordinate'][0]
                cell_z = cell_coords_dict['Z Coordinate'][0]
                model_cell.position(cell_x, cell_y, cell_z)

                register_cell(env, pop_name, gid, model_cell)
                num_cells += 1

        h.define_shape()
        if rank == 0:
            logger.info("*** Created %i cells" % num_cells)

        
def make_cell_selection(env):
    """
    Instantiates cell templates for the selected cells according to
    population ranges and NeuroH5 morphology if present.

    :param env: an instance of the `dentate.Env` class
    """
    
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = env.dataset_path
    data_file_path = env.data_file_path
    
    pop_names = env.cell_selection.keys()

    for pop_name in pop_names:
        if rank == 0:
            logger.info("*** Creating selected cells from population %s" % pop_name)
        env.load_cell_template(pop_name)
        templateClass = getattr(h, env.celltypes[pop_name]['template'])

        v_sample_set = set([])
        env.v_sample_dict[pop_name] = v_sample_set

        gid_range = [ gid for gid in env.cell_selection[pop_name] if gid % nhosts == rank ]
        
        for gid in gid_range:
            v_sample_set.add(gid)

        if (pop_name in env.cellAttributeInfo) and ('Trees' in env.cellAttributeInfo[pop_name]):
            if rank == 0:
                logger.info("*** Reading trees for population %s" % pop_name)

            (trees, _) = read_tree_selection(data_file_path, pop_name, gid_range, comm=env.comm)
            if rank == 0:
                logger.info("*** Done reading trees for population %s" % pop_name)

            first_gid = None
            num_cells = 0
            for i, (gid, tree) in enumerate(trees):
                
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))
                if first_gid == None:
                    first_gid = gid
                    
                model_cell = cells.make_neurotree_cell(templateClass, neurotree_dict=tree, gid=gid, 
                                                       dataset_path=dataset_path)
                if rank == 0 and first_gid == gid:
                    for sec in list(model_cell.all):
                        h.psection(sec=sec)

                register_cell(env, pop_name, gid, model_cell)
                if model_cell.is_art() == 0:
                    if gid in env.v_sample_dict[pop_name]: 
                        rec = make_rec(gid, pop_name, gid, model_cell, \
                                       sec=list(model_cell.soma)[0], \
                                       dt=env.dt, loc=0.5, param='v', \
                                       description='Soma')
                        env.recs_dict[pop_name]['Soma'].append(rec)

                num_cells += 1

            if rank == 0:
                logger.info("*** Created %i cells" % num_cells)

        elif (pop_name in env.cellAttributeInfo) and ('Coordinates' in env.cellAttributeInfo[pop_name]):
            if rank == 0:
                logger.info("*** Reading coordinates for population %s" % pop_name)

            cell_attributes_iter = read_cell_attribute_selection(data_file_path, pop_name, selection=gid_range, \
                                                                 namespace='Coordinates', comm=env.comm)

            if rank == 0:
                logger.info("*** Done reading coordinates for population %s" % pop_name)

            i = 0
            for (gid, cell_coords_dict) in cell_attributes_iter:
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))

                model_cell = cells.make_hoc_cell(env, pop_name, gid)

                cell_x = cell_coords_dict['X Coordinate'][0]
                cell_y = cell_coords_dict['Y Coordinate'][0]
                cell_z = cell_coords_dict['Z Coordinate'][0]
                model_cell.position(cell_x, cell_y, cell_z)
                register_cell(env, pop_name, gid, model_cell)
                if model_cell.is_art() == 0:
                    if gid in env.v_sample_dict[pop_name]:
                        rec = make_rec(gid, pop_name, gid, model_cell, \
                                       sec=list(model_cell.soma)[0], \
                                       dt=env.dt, loc=0.5, param='v', \
                                       description='Soma') 
                        env.recs_dict[pop_name]['Soma'].append(rec)

                i = i + 1
        h.define_shape()


def make_stimulus(env):
    """
    Loads spike train data from NeuroH5 file for those populations
    that have 'Vector Stimulus' entry in the cell configuration.

    :param env: an instance of the `dentate.Env` class
    """
    
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = env.dataset_path
    input_file_path = env.data_file_path

    pop_names = list(env.celltypes.keys())
    pop_names.sort()
    for pop_name in pop_names:
        if 'Vector Stimulus' in env.celltypes[pop_name]:
            vecstim_namespace = env.celltypes[pop_name]['Vector Stimulus']

            if env.node_ranks is None:
                cell_attributes_dict = scatter_read_cell_attributes(input_file_path, pop_name,
                                                                    namespaces=[vecstim_namespace],
                                                                    comm=env.comm, io_size=env.io_size)
            else:
                cell_attributes_dict = scatter_read_cell_attributes(input_file_path, pop_name,
                                                                    namespaces=[vecstim_namespace],
                                                                    node_rank_map=env.node_ranks,
                                                                    comm=env.comm, io_size=env.io_size)
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


def make_stimulus_selection(env, vecstim_sources):
    """
    Loads spike train data from NeuroH5 file for those populations
    that have 'Vector Stimulus' entry in the cell configuration.

    :param env: an instance of the `dentate.Env` class
    :param vecstim_sources: a dictionary of the form { pop_name, gid_range_stim }
    If provided, the set of gids specified in gid_range_stim will be instantiated as VecStims
    and their spike trains will be loaded from the file and namespace specified in 
    env.spike_input_path and env.spike_input_ns, respectively.
    """
    
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    dataset_path = env.dataset_path
    input_file_path = env.data_file_path

    pop_names = list(env.celltypes.keys())
    pop_names.sort()
    for pop_name in pop_names:
        if 'Vector Stimulus' in env.celltypes[pop_name]:
            vecstim_namespace = env.celltypes[pop_name]['Vector Stimulus']

            gid_range = [ gid for gid in env.cell_selection[pop_name] if gid % nhosts == rank ]

            cell_vecstim_iter = read_cell_attribute_selection(input_file_path, pop_name, gid_range, \
                                                              namespace=vecstim_namespace, \
                                                              comm=env.comm)

            if rank == 0:
                logger.info("*** Stimulus onset is %g ms" % env.stimulus_onset)
            for (gid, vecstim_dict) in cell_vecstim_iter:
                if len(vecstim_dict['spiketrain']) > 0:
                    logger.info("*** Spike train for gid %i is of length %i (first spike at %g ms)" %
                                (gid, len(vecstim_dict['spiketrain']), vecstim_dict['spiketrain'][0]))
                else:
                    logger.info("*** Spike train for gid %i is of length %i" %
                                (gid, len(vecstim_dict['spiketrain'])))

                vecstim_dict['spiketrain'] += env.stimulus_onset
                cell = env.pc.gid2cell(gid)
                cell.play(h.Vector(vecstim_dict['spiketrain']))
                
            if rank == 0:
                logger.info("*** Initialized stimulus population %s" % pop_name)

    if vecstim_sources is not None:
        if env.spike_input_path is None:
            raise RuntimeError("Spike input path not provided")
        if env.spike_input_ns is None:
            raise RuntimeError("Spike input namespace not provided")
        for pop_name, stim_gid_range in viewitems(vecstim_sources):
            if pop_name in env.cell_selection:
                local_stim_gid_range = stim_gid_range.difference(set(env.cell_selection[pop_name]))
            else:
                local_stim_gid_range = stim_gid_range
            stim_gid_ranges = env.comm.allgather(local_stim_gid_range)
            this_stim_gid_range = []
            for gid_range in stim_gid_ranges:
                for gid in gid_range:
                    if gid % nhosts == rank:
                        this_stim_gid_range.append(gid)

            cell_spikes_iter = read_cell_attribute_selection(env.spike_input_path, pop_name, this_stim_gid_range, \
                                                             namespace=env.spike_input_ns, \
                                                             comm=env.comm)
            for gid, cell_spikes_dict in cell_spikes_iter:
                stim_cell = env.pc.gid2cell(gid)
                stim_cell.play(h.Vector(cell_spikes_dict['t']))

def init(env, cleanup=True):
    """
    Initializes the network by calling make_cells, make_stimulus, connect_cells, connect_gjs.
    If env.optldbal or env.optlptbal are specified, performs load balancing.

    :param env: an instance of the `dentate.Env` class
    """
    from neuron import h
    configure_hoc_env(env)

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if rank == 0:
        io_utils.mkout(env, env.results_file_path)
    if rank == 0:
        logger.info("*** Creating cells...")
    h.startsw()
    env.pc.barrier()
    if env.cell_selection is None:
        make_cells(env)
    else:
        make_cell_selection(env)
    if env.profile_memory and rank == 0:
        profile_memory(logger)
    env.pc.barrier()
    env.mkcellstime = h.stopsw()
    if rank == 0:
        logger.info("*** Cells created in %g seconds" % env.mkcellstime)
    logger.info("*** Rank %i created %i cells" % (rank, len(env.cells)))
    if env.cell_selection is None:
        h.startsw()
        connect_gjs(env)
        env.pc.setup_transfer()
        env.pc.barrier()
        env.connectgjstime = h.stopsw()
        if rank == 0:
            logger.info("*** Gap junctions created in %g seconds" % env.connectgjstime)
            
    h.startsw()
    if env.profile_memory and rank == 0:
        profile_memory(logger)
        
    if env.cell_selection is None:
        connect_cells(env, cleanup)
        vecstim_selection = None
    else:
        vecstim_selection = connect_cell_selection(env)
    env.pc.set_maxstep(10.0)
    env.pc.barrier()
    env.connectcellstime = h.stopsw()
    
    if env.profile_memory and rank == 0:
        profile_memory(logger)

    if rank == 0:
        logger.info("*** Connections created in %g seconds" % env.connectcellstime)
    edge_count = int(sum([env.edge_count[dest] for dest in env.edge_count]))
    logger.info("*** Rank %i created %i connections" % (rank, edge_count))
    h.startsw()
    if env.cell_selection is None:
        make_stimulus(env)
    else:
        make_stimulus_selection(env, vecstim_selection)
    env.mkstimtime = h.stopsw()
    if rank == 0:
        logger.info("*** Stimuli created in %g seconds" % env.mkstimtime)
    h.startsw()
    if env.cell_selection is None:
        for lfp_label,lfp_config_dict in viewitems(env.lfpConfig):
            env.lfp[lfp_label] = \
              lfp.LFP(lfp_label, env.pc, env.celltypes, lfp_config_dict['position'], rho=lfp_config_dict['rho'],
                        dt_lfp=lfp_config_dict['dt'], fdst=lfp_config_dict['fraction'],
                        maxEDist=lfp_config_dict['maxEDist'],
                        seed=int(env.modelConfig['Random Seeds']['Local Field Potential']))
        if rank == 0:
            logger.info("*** LFP objects instantiated")
    lfp_time = h.stopsw()
    setup_time           = env.mkcellstime + env.mkstimtime + env.connectcellstime + env.connectgjstime + lfp_time
    max_setup_time       = env.pc.allreduce(setup_time, 2) ## maximum value
    env.simtime          = simtime.SimTimeEvent(env.pc, env.max_walltime_hours, env.results_write_time, max_setup_time)
    h.v_init = env.v_init
    h.stdinit()
    if env.coredat:
        env.pc.nrnbbcore_write("dentate.coredat")
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)


def run(env, output=True, shutdown=True):
    """
    Runs network simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env: an instance of the `dentate.Env` class
    :param output: if True, output spike and cell voltage trace data
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if rank == 0:
        logger.info("*** Running simulation")
        
    env.t_vec.resize(0)
    env.id_vec.resize(0)
    
    h.t = 0
    h.tstop = env.tstop

    h.finitialize(env.v_init)
    
    env.pc.barrier()
    env.pc.psolve(h.tstop)

    if rank == 0:
        logger.info("*** Simulation completed")

    if shutdown:
        del env.cells

    env.pc.barrier()
    if output:
        if rank == 0:
            logger.info("*** Writing spike data")
        io_utils.spikeout(env, env.results_file_path)
        if env.vrecord_fraction > 0.:
          if rank == 0:
            logger.info("*** Writing intracellular trace data")
          t_vec = np.arange(0, h.tstop+h.dt, h.dt, dtype=np.float32)
          io_utils.recsout(env, env.results_file_path)
        env.pc.barrier()
        if rank == 0:
            logger.info("*** Writing local field potential data")
            io_utils.lfpout(env, env.results_file_path)

    comptime = env.pc.step_time()
    cwtime   = comptime + env.pc.step_wait()
    maxcw    = env.pc.allreduce(cwtime, 2)
    meancomp = env.pc.allreduce(comptime, 1)/nhosts
    maxcomp  = env.pc.allreduce(comptime, 2)

    gjtime   = env.pc.vtransfer_time()

    gjvect   = h.Vector()
    env.pc.allgather(gjtime, gjvect)
    meangj = gjvect.mean()
    maxgj  = gjvect.max()
    
    if rank == 0:
        logger.info("Execution time summary for host %i:" % rank)
        logger.info("  created cells in %g seconds" % env.mkcellstime)
        logger.info("  connected cells in %g seconds" % env.connectcellstime)
        logger.info("  created gap junctions in %g seconds" % env.connectgjstime)
        logger.info("  ran simulation in %g seconds" % comptime)
        logger.info("  spike communication time: %g seconds" % env.pc.send_time())
        logger.info("  event handling time: %g seconds" % env.pc.event_time())
        logger.info("  numerical integration time: %g seconds" % env.pc.integ_time())
        logger.info("  voltage transfer time: %g seconds" % gjtime)
        if maxcw > 0:
            logger.info("Load balance = %g" % (meancomp/maxcw))
        if meangj > 0:
            logger.info("Mean/max voltage transfer time: %g / %g seconds" % (meangj, maxgj))
            for i in range(nhosts):
                logger.info("Voltage transfer time on host %i: %g seconds" % (i, gjvect.x[i]))

    if shutdown:
        env.pc.runworker()
        env.pc.done()
        h.quit()
