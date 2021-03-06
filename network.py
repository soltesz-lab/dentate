"""
Dentate Gyrus network initialization routines.
"""
__author__ = 'See AUTHORS.md'

import os, sys, gc, time
import numpy as np
from mpi4py import MPI

from dentate import cells, io_utils, lfp, lpt, simtime, synapses
from dentate.neuron_utils import h, configure_hoc_env, cx, make_rec, mkgap, load_cell_template
from dentate.utils import compose_iter, imapreduce, get_module_logger, profile_memory, Promise
from dentate.utils import range, str, viewitems, viewkeys, zip, zip_longest
from neuroh5.io import bcast_graph, read_cell_attribute_selection, scatter_read_cell_attribute_selection, read_graph_selection, read_tree_selection, scatter_read_cell_attributes, scatter_read_graph, scatter_read_trees, write_cell_attributes, write_graph

# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)


def set_union(a, b, datatype):
    return a.union(b)

mpi_op_set_union = MPI.Op.Create(set_union, commute=True)


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
        logger.info("*** expected load balance %.2f" % (((sum_cx / nhosts) / max_sum_cx)))


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



def connect_cells(env):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for each postsynaptic cell.

    :param env: an instance of the `dentate.Env` class
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info('*** Connectivity file path is %s' % connectivity_file_path)
        logger.info('*** Reading projections: ')

    biophys_cell_count = 0
    for (postsyn_name, presyn_names) in sorted(viewitems(env.projection_dict)):

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
            
        weight_dicts = []
        has_weights = False
        if 'weights' in synapse_config:
            has_weights = True
            weight_dicts = synapse_config['weights']

        if rank == 0:
            logger.info('*** Reading synaptic attributes of population %s' % (postsyn_name))

        cell_attr_namespaces = ['Synapse Attributes']

        if env.node_allocation is None:
            cell_attributes_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                namespaces=sorted(cell_attr_namespaces),
                                                                comm=env.comm, io_size=env.io_size,
                                                                return_type='tuple')
        else:
            cell_attributes_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                namespaces=sorted(cell_attr_namespaces), 
                                                                comm=env.comm, node_allocation=env.node_allocation,
                                                                io_size=env.io_size,
                                                                return_type='tuple')

        syn_attrs_iter, syn_attrs_info = cell_attributes_dict['Synapse Attributes']
        syn_attrs.init_syn_id_attrs_from_iter(syn_attrs_iter, attr_type='tuple', 
                                              attr_tuple_index=syn_attrs_info, debug=(rank == 0))
        del cell_attributes_dict
        gc.collect()

        weight_attr_mask = list(syn_attrs.syn_mech_names)
        weight_attr_mask.append('syn_id')
        
        if has_weights:
            
            for weight_dict in weight_dicts:

                expr_closure = weight_dict.get('closure', None)
                weights_namespaces = weight_dict['namespace']

                if rank == 0:
                    logger.info('*** Reading synaptic weights of population %s from namespaces %s' % (postsyn_name, str(weights_namespaces)))

                if env.node_allocation is None:
                    weight_attr_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                    namespaces=weights_namespaces, 
                                                                    mask=set(weight_attr_mask),
                                                                    comm=env.comm, io_size=env.io_size,
                                                                    return_type='tuple')
                else:
                    weight_attr_dict = scatter_read_cell_attributes(forest_file_path, postsyn_name,
                                                                    namespaces=weights_namespaces, 
                                                                    mask=set(weight_attr_mask),
                                                                    comm=env.comm, node_allocation=env.node_allocation,
                                                                    io_size=env.io_size, return_type='tuple')
                
                append_weights = False
                multiple_weights = 'error'
                for weights_namespace in weights_namespaces:

                    syn_weights_iter, weight_attr_info = weight_attr_dict[weights_namespace]
                    first_gid = None
                    syn_id_index = weight_attr_info.get('syn_id', None)
                    syn_name_inds = [(syn_name, attr_index) 
                                     for syn_name, attr_index in sorted(viewitems(weight_attr_info)) if syn_name != 'syn_id']
                    for gid, cell_weights_tuple in syn_weights_iter:
                        if first_gid is None:
                            first_gid = gid
                        weights_syn_ids = cell_weights_tuple[syn_id_index]
                        for syn_name, syn_name_index in syn_name_inds:
                            if syn_name not in syn_attrs.syn_mech_names:
                                if rank == 0 and first_gid == gid:
                                    logger.warning('*** connect_cells: population: %s; gid: %i; syn_name: %s '
                                                   'not found in network configuration' %
                                                   (postsyn_name, gid, syn_name))
                            else:
                                weights_values = cell_weights_tuple[syn_name_index]
                                assert(len(weights_syn_ids) == len(weights_values))
                                syn_attrs.add_mech_attrs_from_iter(gid, syn_name,
                                                                   zip_longest(weights_syn_ids,
                                                                               [{'weight': Promise(expr_closure, [x])} for x in weights_values]
                                                                               if expr_closure else [{'weight': x} for x in weights_values]),
                                                                   multiple=multiple_weights, append=append_weights)
                                if rank == 0 and gid == first_gid:
                                    logger.info('*** connect_cells: population: %s; gid: %i; found %i %s synaptic weights (%s)' %
                                                (postsyn_name, gid, len(weights_values), syn_name, weights_namespace))
                    expr_closure = None
                    append_weights = True
                    multiple_weights='overwrite'
                    del weight_attr_dict[weights_namespace]


        env.edge_count[postsyn_name] = 0
        for presyn_name in presyn_names:
            env.comm.barrier()
            if rank == 0:
                logger.info('Rank %i: Reading projection %s -> %s' % (rank, presyn_name, postsyn_name))
            if env.node_allocation is None:
                (graph, a) = scatter_read_graph(connectivity_file_path, comm=env.comm, io_size=env.io_size,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])
            else:
                (graph, a) = scatter_read_graph(connectivity_file_path, comm=env.comm, io_size=env.io_size,
                                                node_allocation=env.node_allocation,
                                                projections=[(presyn_name, postsyn_name)],
                                                namespaces=['Synapses', 'Connections'])
            if rank == 0:
                logger.info('Rank %i: Read projection %s -> %s' % (rank, presyn_name, postsyn_name))
            edge_iter = graph[postsyn_name][presyn_name]

            last_time = time.time()
            
            if env.microcircuit_inputs:
                presyn_input_sources = env.microcircuit_input_sources.get(presyn_name, set([]))
                syn_edge_iter = compose_iter(lambda edgeset: presyn_input_sources.update(edgeset[1][0]), \
                                             edge_iter)
                env.microcircuit_input_sources[presyn_name] = presyn_input_sources
            else:
                syn_edge_iter = edge_iter

            syn_attrs.init_edge_attrs_from_iter(postsyn_name, presyn_name, a, syn_edge_iter)
            if rank == 0:
                logger.info('Rank %i: took %.02f s to initialize edge attributes for projection %s -> %s' % \
                            (rank, time.time() - last_time, presyn_name, postsyn_name))
            del graph[postsyn_name][presyn_name]

            
        first_gid = None
        if postsyn_name in env.biophys_cells:
            biophys_cell_count += len(env.biophys_cells[postsyn_name])
            for gid in env.biophys_cells[postsyn_name]:
                if env.node_allocation is not None:
                    assert(gid in env.node_allocation)
                if first_gid is None:
                    first_gid = gid
                try:
                    biophys_cell = env.biophys_cells[postsyn_name][gid]
                    cells.init_biophysics(biophys_cell, env=env, 
                                          reset_cable=True, 
                                          correct_cm=correct_for_spines,
                                          correct_g_pas=correct_for_spines, 
                                          verbose=((rank == 0) and (first_gid == gid)))
                    synapses.init_syn_mech_attrs(biophys_cell, env)
                except KeyError:
                    raise KeyError('*** connect_cells: population: %s; gid: %i; could not initialize biophysics' %
                                     (postsyn_name, gid))

    gc.collect()

    ##
    ## This section instantiates cells that are not part of the
    ## network, but are presynaptic sources for cells that _are_
    ## part of the network. It is necessary to create cells at
    ## this point, as NEURON's ParallelContext does not allow the
    ## creation of gids after netcons including those gids are
    ## created.
    ##
    if env.microcircuit_inputs:
        make_input_cell_selection(env)
    gc.collect()

    first_gid = None
    start_time = time.time()

    gids = list(syn_attrs.gids())
    comm0 = env.comm.Split(2 if len(gids) > 0 else 0, 0)

    assert(len(gids) == biophys_cell_count)
    for gid in gids:
        if first_gid is None:
            first_gid = gid
        if not env.pc.gid_exists(gid):
            logger.info("connect_cells: rank %i: gid %d does not exist" % (rank, gid))
        assert(gid in env.gidset)
        assert(env.pc.gid_exists(gid))
        postsyn_cell = env.pc.gid2cell(gid)
        postsyn_name = find_gid_pop(env.celltypes, gid)

        if rank == 0 and gid == first_gid:
            logger.info('Rank %i: configuring synapses for gid %i' % (rank, gid))

        last_time = time.time()
        
        is_reduced = False
        if hasattr(postsyn_cell, 'is_reduced'):
            is_reduced = postsyn_cell.is_reduced
        syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(
            env, gid, postsyn_name, cell=postsyn_cell.hoc_cell if is_reduced else postsyn_cell, 
            unique=unique, insert=True, insert_netcons=True)

        if rank == 0 and gid == first_gid:
            logger.info('Rank %i: took %.02f s to configure %i synapses, %i synaptic mechanisms, %i network '
                        'connections for gid %d' % \
                        (rank, time.time() - last_time, syn_count, mech_count, nc_count, gid))

        env.edge_count[postsyn_name] += syn_count

        if gid in env.recording_sets.get(postsyn_name, {}):
            cells.record_cell(env, postsyn_name, gid)

        if env.cleanup:
            syn_attrs.del_syn_id_attr_dict(gid)
            if gid in env.biophys_cells[postsyn_name]:
                del env.biophys_cells[postsyn_name][gid]

    comm0.Free()
    gc.collect()

    if rank == 0:
        logger.info('Rank %i: took %.02f s to configure all synapses' %
                    (rank, time.time() - start_time))


def find_gid_pop(celltypes, gid):
    """
    Given a celltypes structure and a gid, find the population to which the gid belongs.
    """
    for pop_name in celltypes:
        start = celltypes[pop_name]['start']
        num = celltypes[pop_name]['num']
        if (start <= gid) and (gid < (start + num)):
            return pop_name

    return None


def connect_cell_selection(env):
    """
    Loads NeuroH5 connectivity file, instantiates the corresponding
    synapse and network connection mechanisms for the selected postsynaptic cells.

    :param env: an instance of the `dentate.Env` class
    """
    connectivity_file_path = env.connectivity_file_path
    forest_file_path = env.forest_file_path
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    syn_attrs = env.synapse_attributes

    if rank == 0:
        logger.info('*** Connectivity file path is %s' % connectivity_file_path)
        logger.info('*** Reading projections: ')

    selection_pop_names = sorted(viewkeys(env.cell_selection))
    biophys_cell_count = 0
    for postsyn_name in sorted(viewkeys(env.projection_dict)):

        if rank == 0:
            logger.info('*** Postsynaptic population: %s' % postsyn_name)

        if postsyn_name not in selection_pop_names:
            continue

        presyn_names = sorted(env.projection_dict[postsyn_name])


        gid_range = [gid for gid in env.cell_selection[postsyn_name] if env.pc.gid_exists(gid)]

        synapse_config = env.celltypes[postsyn_name]['synapses']
        if 'correct_for_spines' in synapse_config:
            correct_for_spines = synapse_config['correct_for_spines']
        else:
            correct_for_spines = False

        if 'unique' in synapse_config:
            unique = synapse_config['unique']
        else:
            unique = False

        weight_dicts = []
        has_weights = False
        if 'weights' in synapse_config:
            has_weights = True
            weight_dicts = synapse_config['weights']

        if rank == 0:
            logger.info('*** Reading synaptic attributes of population %s' % (postsyn_name))

        syn_attrs_iter, syn_attrs_info = read_cell_attribute_selection(forest_file_path, postsyn_name, selection=gid_range,
                                                                       namespace='Synapse Attributes', comm=env.comm, 
                                                                       return_type='tuple')

        syn_attrs.init_syn_id_attrs_from_iter(syn_attrs_iter, attr_type='tuple', attr_tuple_index=syn_attrs_info)
        del (syn_attrs_iter)

        weight_attr_mask = list(syn_attrs.syn_mech_names)
        weight_attr_mask.append('syn_id')

        if has_weights:

            for weight_dict in weight_dicts:

                expr_closure = weight_dict.get('closure', None)
                weights_namespaces = weight_dict['namespace']

                if rank == 0:
                    logger.info('*** Reading synaptic weights of population %s from namespaces %s' % (postsyn_name, str(weights_namespaces)))
                    
                append_weights = False
                multiple_weights='error'

                for weights_namespace in weights_namespaces:
                
                    syn_weights_iter, weight_attr_info = read_cell_attribute_selection(forest_file_path, postsyn_name,
                                                                                   selection=gid_range, 
                                                                                   mask=set(weight_attr_mask),
                                                                                   namespace=weights_namespace, 
                                                                                   comm=env.comm, return_type='tuple')

                    first_gid = None
                    syn_id_index = weight_attr_info.get('syn_id', None)
                    syn_name_inds = [(syn_name, attr_index) for syn_name, attr_index in sorted(viewitems(weight_attr_info)) if syn_name != 'syn_id']

                    for gid, cell_weights_tuple in syn_weights_iter:
                        if first_gid is None:
                            first_gid = gid
                        weights_syn_ids = cell_weights_tuple[syn_id_index]
                        for syn_name, syn_name_index in syn_name_inds:
                            if syn_name not in syn_attrs.syn_mech_names:
                                if rank == 0 and first_gid == gid:
                                    logger.warning('*** connect_cells: population: %s; gid: %i; syn_name: %s '
                                                'not found in network configuration' %
                                                (postsyn_name, gid, syn_name))
                            else:
                                weights_values = cell_weights_tuple[syn_name_index]
                                syn_attrs.add_mech_attrs_from_iter(gid, syn_name,
                                                                   zip_longest(weights_syn_ids,
                                                                               [{'weight': Promise(expr_closure, [x])} for x in weights_values]
                                                                               if expr_closure else [{'weight': x} for x in weights_values]),
                                                                   multiple=multiple_weights, append=append_weights)

                                if rank == 0 and gid == first_gid:
                                    logger.info('*** connect_cells: population: %s; gid: %i; found %i %s synaptic weights (%s)' %
                                            (postsyn_name, gid, len(weights_values), syn_name, weights_namespace))
                multiple_weights='overwrite'
                append_weights=True
                del syn_weights_iter

                
        (graph, a) = read_graph_selection(connectivity_file_path, selection=gid_range, \
                                          projections=[ (presyn_name, postsyn_name) for presyn_name in sorted(presyn_names) ],
                                          comm=env.comm, namespaces=['Synapses', 'Connections'])

        env.edge_count[postsyn_name] = 0
        if postsyn_name in graph:
            for presyn_name in presyn_names:

                logger.info('*** Connecting %s -> %s' % (presyn_name, postsyn_name))

                edge_iter = graph[postsyn_name][presyn_name]
                
                presyn_input_sources = env.microcircuit_input_sources.get(presyn_name, set([]))
                syn_edge_iter = compose_iter(lambda edgeset: presyn_input_sources.update(edgeset[1][0]),
                                             edge_iter)
                syn_attrs.init_edge_attrs_from_iter(postsyn_name, presyn_name, a, syn_edge_iter)
                env.microcircuit_input_sources[presyn_name] = presyn_input_sources
                del graph[postsyn_name][presyn_name]


        first_gid = None
        if postsyn_name in env.biophys_cells:
            biophys_cell_count += len(env.biophys_cells[postsyn_name])
            for gid in env.biophys_cells[postsyn_name]:
                if env.node_allocation is not None:
                    assert(gid in env.node_allocation)
                if first_gid is None:
                    first_gid = gid
                try:
                    if syn_attrs.has_gid(gid):
                        biophys_cell = env.biophys_cells[postsyn_name][gid]
                        cells.init_biophysics(biophys_cell,
                                              reset_cable=True,
                                              correct_cm=correct_for_spines,
                                              correct_g_pas=correct_for_spines,
                                              env=env, verbose=((rank == 0) and (first_gid == gid)))
                        synapses.init_syn_mech_attrs(biophys_cell, env)
                except KeyError:
                    raise KeyError('connect_cells: population: %s; gid: %i; could not initialize biophysics'
                                     % (postsyn_name, gid))


    ##
    ## This section instantiates cells that are not part of the
    ## selection, but are presynaptic sources for cells that _are_
    ## part of the selection. It is necessary to create cells at this
    ## point, as NEURON's ParallelContext does not allow the creation
    ## of gids after netcons including those gids are created.
    ##
    make_input_cell_selection(env)

    ##
    ## This section instantiates the synaptic mechanisms and netcons for each connection.
    ##
    first_gid = None
    gids = list(syn_attrs.gids())
    assert(len(gids) == biophys_cell_count)

    for gid in gids:

        last_time = time.time()
        if first_gid is None:
            first_gid = gid

        cell = env.pc.gid2cell(gid)
        pop_name = find_gid_pop(env.celltypes, gid)

        if hasattr(cell, 'is_reduced'):
            is_reduced = cell.is_reduced

        syn_count, mech_count, nc_count = synapses.config_hoc_cell_syns(
            env, gid, pop_name, cell=cell.hoc_cell if is_reduced else cell,
            unique=unique, insert=True, insert_netcons=True)

        if rank == 0 and gid == first_gid:
            logger.info('Rank %i: took %.02f s to configure %i synapses, %i synaptic mechanisms, %i network '
                        'connections for gid %d; cleanup flag is %s' % \
                        (rank, time.time() - last_time, syn_count, mech_count, nc_count, gid, str(env.cleanup)))
            hoc_cell = env.pc.gid2cell(gid)
            if hasattr(hoc_cell, 'all'):
                for sec in list(hoc_cell.all):
                    h.psection(sec=sec)

        if gid in env.recording_sets.get(pop_name, {}):
            cells.record_cell(env, pop_name, gid)

        env.edge_count[pop_name] += syn_count
        if env.cleanup:
            syn_attrs.del_syn_id_attr_dict(gid)
            if gid in env.biophys_cells[pop_name]:
                del env.biophys_cells[pop_name][gid]



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

        (graph, a) = bcast_graph(gapjunctions_file_path, \
                                 namespaces=['Coupling strength', 'Location'], \
                                 comm=env.comm)

        ggid = 2e6
        for name in sorted(viewkeys(gapjunctions)):
            if rank == 0:
                logger.info("*** Creating gap junctions %s" % str(name))
            prj = graph[name[0]][name[1]]
            attrmap = a[(name[1], name[0])]
            cc_src_idx = attrmap['Coupling strength']['Source']
            cc_dst_idx = attrmap['Coupling strength']['Destination']
            dstsec_idx = attrmap['Location']['Destination section']
            dstpos_idx = attrmap['Location']['Destination position']
            srcsec_idx = attrmap['Location']['Source section']
            srcpos_idx = attrmap['Location']['Source position']

            for src in sorted(viewkeys(prj)):
                edges = prj[src]
                destinations = edges[0]
                cc_dict = edges[1]['Coupling strength']
                loc_dict = edges[1]['Location']
                srcweights = cc_dict[cc_src_idx]
                dstweights = cc_dict[cc_dst_idx]
                dstposs = loc_dict[dstpos_idx]
                dstsecs = loc_dict[dstsec_idx]
                srcposs = loc_dict[srcpos_idx]
                srcsecs = loc_dict[srcsec_idx]
                for i in range(0, len(destinations)):
                    dst = destinations[i]
                    srcpos = srcposs[i]
                    srcsec = srcsecs[i]
                    dstpos = dstposs[i]
                    dstsec = dstsecs[i]
                    wgt = srcweights[i] * 0.001
                    if env.pc.gid_exists(src):
                        if rank == 0:
                            logger.info('host %d: gap junction: gid = %d sec = %d coupling = %g '
                                        'sgid = %d dgid = %d\n' %
                                        (rank, src, srcsec, wgt, ggid, ggid + 1))
                        cell = env.pc.gid2cell(src)
                        gj = mkgap(env, cell, src, srcpos, srcsec, ggid, ggid + 1, wgt)
                    if env.pc.gid_exists(dst):
                        if rank == 0:
                            logger.info('host %d: gap junction: gid = %d sec = %d coupling = %g '
                                        'sgid = %d dgid = %d\n' %
                                        (rank, dst, dstsec, wgt, ggid + 1, ggid))
                        cell = env.pc.gid2cell(dst)
                        gj = mkgap(env, cell, dst, dstpos, dstsec, ggid + 1, ggid, wgt)
                    ggid = ggid + 2
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

    recording_seed = int(env.model_config['Random Seeds']['Intracellular Recording Sample'])
    ranstream_recording = np.random.RandomState()
    ranstream_recording.seed(recording_seed)

    dataset_path = env.dataset_path
    data_file_path = env.data_file_path
    pop_names = sorted(viewkeys(env.celltypes))

    if rank == 0:
        logger.info("Population attributes: %s" % str(env.cell_attribute_info))
    for pop_name in pop_names:
        if rank == 0:
            logger.info("*** Creating population %s" % pop_name)
            
        
        template_name = env.celltypes[pop_name].get('template', None)
        if template_name is None:
            continue
        
        template_name_lower = template_name.lower()
        if template_name_lower != 'izhikevich' and template_name_lower != 'vecstim':    
            load_cell_template(env, pop_name, bcast_template=True)

                
        if 'mech_file_path' in env.celltypes[pop_name]:
            mech_dict = env.celltypes[pop_name]['mech_dict']
            mech_file_path = env.celltypes[pop_name]['mech_file_path']
            if rank == 0:
                logger.info('*** Mechanism file for population %s is %s' % (pop_name, str(mech_file_path)))
        else:
            mech_dict = None

        is_reduced = (template_name.lower() == 'izhikevich')
        num_cells = 0
        if (pop_name in env.cell_attribute_info) and ('Trees' in env.cell_attribute_info[pop_name]):
            if rank == 0:
                logger.info("*** Reading trees for population %s" % pop_name)

            if env.node_allocation is None:
                (trees, forestSize) = scatter_read_trees(data_file_path, pop_name, comm=env.comm, \
                                                         io_size=env.io_size)
            else:
                (trees, forestSize) = scatter_read_trees(data_file_path, pop_name, comm=env.comm, \
                                                         io_size=env.io_size, node_allocation=env.node_allocation)
            if rank == 0:
                logger.info("*** Done reading trees for population %s" % pop_name)

            first_gid = None
            for i, (gid, tree) in enumerate(trees):
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))

                if first_gid is None:
                    first_gid = gid

                if is_reduced:
                    izhikevich_cell = cells.make_izhikevich_cell(gid=gid, pop_name=pop_name,
                                                                 env=env, mech_dict=mech_dict)
                    cells.register_cell(env, pop_name, gid, izhikevich_cell)
                else:
                    hoc_cell = cells.make_hoc_cell(env, pop_name, gid, neurotree_dict=tree)
                    if mech_dict is None:
                        cells.register_cell(env, pop_name, gid, hoc_cell)
                    else:
                        biophys_cell = cells.make_biophys_cell(gid=gid, pop_name=pop_name,
                                                               hoc_cell=hoc_cell, env=env,
                                                               tree_dict=tree,
                                                               mech_dict=mech_dict)
                        # cells.init_spike_detector(biophys_cell)
                        cells.register_cell(env, pop_name, gid, biophys_cell)
                        if rank == 0 and gid == first_gid:
                            logger.info('*** make_cells: population: %s; gid: %i; loaded biophysics from path: %s' %
                                        (pop_name, gid, mech_file_path))

                    if rank == 0 and first_gid == gid:
                        if hasattr(hoc_cell, 'all'):
                            for sec in list(hoc_cell.all):
                                h.psection(sec=sec)

                num_cells += 1
            del trees

            
        elif (pop_name in env.cell_attribute_info) and ('Coordinates' in env.cell_attribute_info[pop_name]):
            if rank == 0:
                logger.info("*** Reading coordinates for population %s" % pop_name)

            if env.node_allocation is None:
                cell_attr_dict = scatter_read_cell_attributes(data_file_path, pop_name,
                                                              namespaces=['Coordinates'],
                                                              comm=env.comm, io_size=env.io_size,
                                                              return_type='tuple')
            else:
                cell_attr_dict = scatter_read_cell_attributes(data_file_path, pop_name,
                                                              namespaces=['Coordinates'],
                                                              node_allocation=env.node_allocation,
                                                              comm=env.comm, io_size=env.io_size,
                                                              return_type='tuple')
            if rank == 0:
                logger.info("*** Done reading coordinates for population %s" % pop_name)

            coords_iter, coords_attr_info = cell_attr_dict['Coordinates']

            x_index = coords_attr_info.get('X Coordinate', None)
            y_index = coords_attr_info.get('Y Coordinate', None)
            z_index = coords_attr_info.get('Z Coordinate', None)
            for i, (gid, cell_coords) in enumerate(coords_iter):
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))

                if is_reduced:
                    izhikevich_cell = cells.make_izhikevich_cell(gid=gid, pop_name=pop_name,
                                                                 env=env, mech_dict=mech_dict)
                    cells.register_cell(env, pop_name, gid, izhikevich_cell)
                else:
                    hoc_cell = cells.make_hoc_cell(env, pop_name, gid)
                    cell_x = cell_coords[x_index][0]
                    cell_y = cell_coords[y_index][0]
                    cell_z = cell_coords[z_index][0]
                    hoc_cell.position(cell_x, cell_y, cell_z)
                    cells.register_cell(env, pop_name, gid, hoc_cell)
                num_cells += 1
        else:
            raise RuntimeError("make_cells: unknown cell configuration type for cell type %s" % pop_name)

        h.define_shape()

        recording_set = set([])
        pop_biophys_gids = list(env.biophys_cells[pop_name].keys())
        pop_biophys_gids_per_rank = env.comm.gather(pop_biophys_gids, root=0)
        if rank == 0:
            all_pop_biophys_gids = sorted([item for sublist in pop_biophys_gids_per_rank for item in sublist])
            for gid in all_pop_biophys_gids:
                if ranstream_recording.uniform() <= env.recording_fraction:
                    recording_set.add(gid)
        recording_set = env.comm.bcast(recording_set, root=0)
        env.recording_sets[pop_name] = recording_set
        del pop_biophys_gids_per_rank
        logger.info("*** Rank %i: Created %i cells from population %s" % (rank, num_cells, pop_name))

    # if node rank map has not been created yet, create it now
    if env.node_allocation is None:
        env.node_allocation = set([])
        for gid in env.gidset:
            env.node_allocation.add(gid)


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

    pop_names = sorted(viewkeys(env.cell_selection))

    for pop_name in pop_names:
        if rank == 0:
            logger.info("*** Creating selected cells from population %s" % pop_name)

        template_name = env.celltypes[pop_name]['template']
        template_name_lower = template_name.lower()
        if template_name_lower != 'izhikevich' and template_name_lower != 'vecstim':    
            load_cell_template(env, pop_name, bcast_template=True)

        templateClass = getattr(h, env.celltypes[pop_name]['template'])

        gid_range = [gid for gid in env.cell_selection[pop_name] if gid % nhosts == rank]

        if 'mech_file_path' in env.celltypes[pop_name]:
            mech_dict = env.celltypes[pop_name]['mech_dict']
        else:
            mech_dict = None

        is_reduced = (template_name.lower() == 'izhikevich')
        num_cells = 0
        
            
        if (pop_name in env.cell_attribute_info) and ('Trees' in env.cell_attribute_info[pop_name]):
            if rank == 0:
                logger.info("*** Reading trees for population %s" % pop_name)

            (trees, _) = read_tree_selection(data_file_path, pop_name, gid_range, comm=env.comm)
            if rank == 0:
                logger.info("*** Done reading trees for population %s" % pop_name)

            first_gid = None
            for i, (gid, tree) in enumerate(trees):

                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))
                if first_gid == None:
                    first_gid = gid

                if is_reduced:
                    izhikevich_cell = cells.make_izhikevich_cell(gid=gid, pop_name=pop_name,
                                                                 env=env, param_dict=mech_dict)
                    cells.register_cell(env, pop_name, gid, izhikevich_cell)
                else:
                    hoc_cell = cells.make_hoc_cell(env, pop_name, gid, neurotree_dict=tree)
                    if mech_file_path is None:
                        cells.register_cell(env, pop_name, gid, hoc_cell)
                    else:
                        biophys_cell = cells.make_biophys_cell(gid=gid, pop_name=pop_name,
                                                               hoc_cell=hoc_cell, env=env,
                                                               tree_dict=tree,
                                                               mech_dict=mech_dict)
                        # cells.init_spike_detector(biophys_cell)
                        cells.register_cell(env, pop_name, gid, biophys_cell)
                        if rank == 0 and gid == first_gid:
                            logger.info('*** make_cell_selection: population: %s; gid: %i; loaded biophysics from path: %s' %
                                        (pop_name, gid, mech_file_path))
                            

                if rank == 0 and first_gid == gid:
                    if hasattr(hoc_cell, 'all'):
                        for sec in list(hoc_cell.all):
                            h.psection(sec=sec)

                num_cells += 1
                

        elif (pop_name in env.cell_attribute_info) and ('Coordinates' in env.cell_attribute_info[pop_name]):
            if rank == 0:
                logger.info("*** Reading coordinates for population %s" % pop_name)

            coords_iter, coords_attr_info = read_cell_attribute_selection(data_file_path, pop_name, selection=gid_range, 
                                                                          namespace='Coordinates', comm=env.comm, 
                                                                          return_type='tuple')
            x_index = coords_attr_info.get('X Coordinate', None)
            y_index = coords_attr_info.get('Y Coordinate', None)
            z_index = coords_attr_info.get('Z Coordinate', None)

            if rank == 0:
                logger.info("*** Done reading coordinates for population %s" % pop_name)

            for i, (gid, cell_coords_tuple) in enumerate(coords_iter):
                if rank == 0:
                    logger.info("*** Creating %s gid %i" % (pop_name, gid))

                if is_reduced:
                    izhikevich_cell = cells.make_izhikevich_cell(gid=gid, pop_name=pop_name,
                                                                 env=env, param_dict=mech_dict)
                    cells.register_cell(env, pop_name, gid, izhikevich_cell)
                else:
                    hoc_cell = cells.make_hoc_cell(env, pop_name, gid)

                    cell_x = cell_coords_tuple[x_index][0]
                    cell_y = cell_coords_tuple[y_index][0]
                    cell_z = cell_coords_tuple[z_index][0]
                    hoc_cell.position(cell_x, cell_y, cell_z)
                    cells.register_cell(env, pop_name, gid, hoc_cell)

                num_cells += 1

        h.define_shape()
        logger.info("*** Rank %i: Created %i cells from population %s" % (rank, num_cells, pop_name))


    if env.node_allocation is None:
        env.node_allocation = set([])
        for gid in env.gidset:
            env.node_allocation.add(gid)


def make_input_cell_selection(env):
    """
    Creates cells with predefined spike patterns when only a subset of the network is instantiated.

    :param env: an instance of the `dentate.Env` class
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    created_input_sources = { pop_name: set([]) for pop_name in env.celltypes.keys() }
    for pop_name, input_gid_range in sorted(viewitems(env.microcircuit_input_sources)):

        pop_index = int(env.Populations[pop_name])

        has_spike_train = False
        if (env.spike_input_attribute_info is not None) and (env.spike_input_ns is not None):
            if (pop_name in env.spike_input_attribute_info) and \
              (env.spike_input_ns in env.spike_input_attribute_info[pop_name]):
                has_spike_train = True

        if has_spike_train:
            spike_generator = None
        else:
            if env.netclamp_config is None:
                logger.warning("make_input_cell_selection: population %s has neither input spike trains nor input generator configuration" % pop_name)
                spike_generator = None
            else:
                if pop_name in env.netclamp_config.input_generators:
                    spike_generator = env.netclamp_config.input_generators[pop_name]
                else:
                    raise RuntimeError("make_input_cell_selection: population %s has neither input spike trains nor input generator configuration" % pop_name)

        if spike_generator is not None:
            input_source_dict = {pop_index: {'generator': spike_generator}}
        else:
            input_source_dict = {pop_index: {'spiketrains': {}}}


        if (env.cell_selection is not None) and (pop_name in env.cell_selection):
            local_input_gid_range = input_gid_range.difference(set(env.cell_selection[pop_name]))
        else:
            local_input_gid_range = input_gid_range
        input_gid_ranges = env.comm.allreduce(local_input_gid_range, op=mpi_op_set_union)

        created_input_gids = []
        for i, gid in enumerate(input_gid_ranges):
            if (i % nhosts == rank) and not env.pc.gid_exists(gid):
                input_cell = cells.make_input_cell(env, gid, pop_index, input_source_dict)
                cells.register_cell(env, pop_name, gid, input_cell)
                created_input_gids.append(gid)
        created_input_sources[pop_name] = set(created_input_gids)

        if rank == 0:
            logger.info('*** Rank %i: created %s input sources for gids %s' % (rank, pop_name, str(created_input_gids)))

    env.microcircuit_input_sources = created_input_sources

    if env.node_allocation is None:
        env.node_allocation = set([])
    for _, this_gidset in viewitems(env.microcircuit_input_sources):
        for gid in this_gidset:
            env.node_allocation.add(gid)

def merge_spiketrain_trials(spiketrain, trial_index, trial_duration, n_trials):
    if trial_index is not None:
        trial_spiketrains = []
        for trial_i in range(n_trials):
            trial_spiketrain_i = spiketrain[np.where(trial_index == trial_i)[0]]
            trial_spiketrain_i += np.sum(trial_duration[:trial_i])
            trial_spiketrains.append(trial_spiketrain_i)
            spiketrain = np.concatenate(trial_spiketrains)
    return np.sort(spiketrain)

def init_input_cells(env):
    """
    Initializes cells with predefined spike patterns.

    :param env: an instance of the `dentate.Env` class
    """

    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    if rank == 0:
        logger.info("*** Stimulus onset is %g ms" % env.stimulus_onset)

    dataset_path = env.dataset_path
    input_file_path = env.data_file_path

    pop_names = sorted(viewkeys(env.celltypes))

    trial_index_attr = 'Trial Index'
    trial_dur_attr = 'Trial Duration'
    for pop_name in pop_names:

        if 'spike train' in env.celltypes[pop_name]:
            if env.arena_id and env.trajectory_id:
                vecstim_namespace = '%s %s %s' % (env.celltypes[pop_name]['spike train']['namespace'], \
                                                env.arena_id, env.trajectory_id)
            else:
                vecstim_namespace = env.celltypes[pop_name]['spike train']['namespace']
            vecstim_attr = env.celltypes[pop_name]['spike train']['attribute']

            has_vecstim = False
            if env.cell_attribute_info is not None:
                if (pop_name in env.cell_attribute_info) and \
                   (vecstim_namespace in env.cell_attribute_info[pop_name]):
                     has_vecstim = True

            if has_vecstim:
                if rank == 0:
                    logger.info("*** Initializing stimulus population %s from namespace %s" % (pop_name, vecstim_namespace))

                if env.cell_selection is None:
                    if env.node_allocation is None:
                        cell_vecstim_dict = scatter_read_cell_attributes(input_file_path, pop_name,
                                                                         namespaces=[vecstim_namespace],
                                                                         mask=set([vecstim_attr, trial_index_attr, trial_dur_attr]),
                                                                         comm=env.comm, io_size=env.io_size,
                                                                         return_type='tuple')
                    else:
                        cell_vecstim_dict = scatter_read_cell_attributes(input_file_path, pop_name,
                                                                         namespaces=[vecstim_namespace],
                                                                         node_allocation=env.node_allocation,
                                                                         mask=set([vecstim_attr, trial_index_attr, trial_dur_attr]),
                                                                         comm=env.comm, io_size=env.io_size,
                                                                         return_type='tuple')
                    vecstim_iter, vecstim_attr_info = cell_vecstim_dict[vecstim_namespace]
                else:
                    if pop_name in env.cell_selection:
                        gid_range = [gid for gid in env.cell_selection[pop_name] if env.pc.gid_exists(gid)]
                        
                        vecstim_iter, vecstim_attr_info = scatter_read_cell_attribute_selection(input_file_path, \
                                                                                                pop_name, gid_range, \
                                                                                                namespace=vecstim_namespace, \
                                                                                                selection=list(gid_range), \
                                                                                                mask=set([vecstim_attr, trial_index_attr, trial_dur_attr]), \
                                                                                                comm=env.comm, io_size=env.io_size, return_type='tuple')
                    else:
                        vecstim_iter = []

                vecstim_attr_index = vecstim_attr_info.get(vecstim_attr, None)
                trial_index_attr_index = vecstim_attr_info.get(trial_index_attr, None)
                trial_dur_attr_index = vecstim_attr_info.get(trial_dur_attr, None)
                for (gid, vecstim_tuple) in vecstim_iter:
                    if not (env.pc.gid_exists(gid)):
                        continue

                    cell = env.artificial_cells[pop_name][gid]

                    spiketrain = vecstim_tuple[vecstim_attr_index]
                    trial_index = None
                    trial_duration = None
                    if trial_index_attr_index is not None:
                        trial_index = vecstim_tuple[trial_index_attr_index]
                        trial_duration = vecstim_tuple[trial_dur_attr_index]
                    if len(spiketrain) > 0:
                        spiketrain = merge_spiketrain_trials(spiketrain, trial_index, trial_duration, env.n_trials)
                        spiketrain += float(env.stimulus_config['Equilibration Duration']) + env.stimulus_onset
                        if len(spiketrain) > 0:
                            cell.play(h.Vector(spiketrain))
                            if rank == 0:
                                logger.info("*** Spike train for %s gid %i is of length %i (%g : %g ms)" %
                                            (pop_name, gid, len(spiketrain), spiketrain[0], spiketrain[-1]))


    gc.collect()

    if env.microcircuit_inputs:

        for pop_name in sorted(viewkeys(env.microcircuit_input_sources)):

            gid_range = env.microcircuit_input_sources.get(pop_name, set([]))

            if (env.cell_selection is not None) and (pop_name in env.cell_selection):
                this_gid_range = gid_range.difference(set(env.cell_selection[pop_name]))
            else:
                this_gid_range = gid_range

            has_spike_train = False
            spike_input_source_loc = []
            if (env.spike_input_attribute_info is not None) and (env.spike_input_ns is not None):
                if (pop_name in env.spike_input_attribute_info) and \
                   (env.spike_input_ns in env.spike_input_attribute_info[pop_name]):
                   has_spike_train = True
                   spike_input_source_loc.append((env.spike_input_path, env.spike_input_ns))
            if (env.cell_attribute_info is not None) and (env.spike_input_ns is not None):
                if (pop_name in env.cell_attribute_info) and \
                        (env.spike_input_ns in env.cell_attribute_info[pop_name]):
                    has_spike_train = True
                    spike_input_source_loc.append((input_file_path, env.spike_input_ns))

            if rank == 0:
                logger.info("*** Initializing input source %s from locations %s" % (pop_name, str(spike_input_source_loc)))
                    
            if has_spike_train:

                vecstim_attr_set = set(['t', trial_index_attr, trial_dur_attr])
                if env.spike_input_attr is not None:
                    vecstim_attr_set.add(env.spike_input_attr)
                if pop_name in env.celltypes:
                    if 'spike train' in env.celltypes[pop_name]:
                        vecstim_attr_set.add(env.celltypes[pop_name]['spike train']['attribute'])
                    
                cell_spikes_items = []
                for (input_path, input_ns) in spike_input_source_loc:
                    item = scatter_read_cell_attribute_selection(
                        input_path, pop_name, list(this_gid_range),
                        namespace=input_ns, mask=vecstim_attr_set,
                        comm=env.comm, io_size=env.io_size, return_type='tuple')
                    cell_spikes_items.append(item)
                    
                for cell_spikes_iter, cell_spikes_attr_info in cell_spikes_items:
                    if len(cell_spikes_attr_info) == 0:
                        continue
                    trial_index_attr_index = cell_spikes_attr_info.get(trial_index_attr, None)
                    trial_dur_attr_index = cell_spikes_attr_info.get(trial_dur_attr, None)
                    if (env.spike_input_attr is not None) and (env.spike_input_attr in cell_spikes_attr_info):
                        spike_train_attr_index = cell_spikes_attr_info.get(env.spike_input_attr, None)
                    elif 't' in viewkeys(cell_spikes_attr_info):
                        spike_train_attr_index = cell_spikes_attr_info.get('t', None)
                    elif 'Spike Train' in viewkeys(cell_spikes_attr_info):
                        spike_train_attr_index = cell_spikes_attr_info.get('Spike Train', None)
                    elif len(this_gid_range) > 0:
                        raise RuntimeError("init_input_cells: unable to determine spike train attribute for population %s in spike input file %s; namespace %s; attr keys %s" % (pop_name, env.spike_input_path, env.spike_input_ns, str(list(viewkeys(cell_spikes_attr_info)))))
                        
                    for gid, cell_spikes_tuple in cell_spikes_iter:
                        if not (env.pc.gid_exists(gid)):
                            continue
                        if gid not in env.artificial_cells[pop_name]:
                            logger.info('init_input_cells: Rank %i: env.artificial_cells[%s] = %s this_gid_range = %s' % (rank, pop_name, str(env.artificial_cells[pop_name]), str(this_gid_range)))
                        input_cell = env.artificial_cells[pop_name][gid]

                        spiketrain = cell_spikes_tuple[spike_train_attr_index]
                        if trial_index_attr_index is None:
                            trial_index = None
                        else:
                            trial_index = cell_spikes_tuple[trial_index_attr_index]
                            trial_duration = cell_spikes_tuple[trial_dur_attr_index]

                        if len(spiketrain) > 0:
                            spiketrain = merge_spiketrain_trials(spiketrain, trial_index, trial_duration, env.n_trials)
                            spiketrain += float(env.stimulus_config['Equilibration Duration']) + env.stimulus_onset

                            input_cell.play(h.Vector(spiketrain))
                            if len(spiketrain) > 0:
                                if rank == 0:
                                    logger.info("*** Spike train for %s input source gid %i is of length %i (%g : %g ms)" %
                                                (pop_name, gid, len(spiketrain), spiketrain[0], spiketrain[-1]))

            else:
                if rank == 0:
                    logger.warning('No spike train data found for population %s in spike input file %s; namespace: %s' % (pop_name, env.spike_input_path, env.spike_input_ns))
            
    gc.collect()
                    

def init(env):
    """
    Initializes the network by calling make_cells, init_input_cells, connect_cells, connect_gjs.
    If env.optldbal or env.optlptbal are specified, performs load balancing.

    :param env: an instance of the `dentate.Env` class
    """
    from neuron import h
    configure_hoc_env(env)
    
    assert(env.data_file_path)
    assert(env.connectivity_file_path)
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())
    if env.optldbal or env.optlptbal:
        lb = h.LoadBalance()
        if not os.path.isfile("mcomplex.dat"):
            lb.ExperimentalMechComplex()

    if rank == 0:
        logger.info("*** Creating cells...")
    st = time.time()

    if env.cell_selection is None:
        make_cells(env)
    else:
        make_cell_selection(env)
    if env.profile_memory and rank == 0:
        profile_memory(logger)

    env.mkcellstime = time.time() - st
    if rank == 0:
        logger.info("*** Cells created in %g seconds" % env.mkcellstime)
    local_num_cells = imapreduce(viewitems(env.cells), lambda kv: len(kv[1]), lambda ax, x: ax+x)
    logger.info("*** Rank %i created %i cells" % (rank, local_num_cells))
    if env.cell_selection is None:
        st = time.time()
        connect_gjs(env)
        env.pc.setup_transfer()
        env.connectgjstime = time.time() - st
        if rank == 0:
            logger.info("*** Gap junctions created in %g seconds" % env.connectgjstime)

    if env.profile_memory and rank == 0:
        profile_memory(logger)

    st = time.time()
    if (not env.use_coreneuron) and (len(env.LFP_config) > 0):
        lfp_pop_dict = { pop_name: set(viewkeys(env.cells[pop_name])).difference(set(viewkeys(env.artificial_cells[pop_name])))
                         for pop_name in viewkeys(env.cells) }
        for lfp_label, lfp_config_dict in sorted(viewitems(env.LFP_config)):
            env.lfp[lfp_label] = lfp.LFP(lfp_label, env.pc, lfp_pop_dict,
                                         lfp_config_dict['position'], rho=lfp_config_dict['rho'],
                                         dt_lfp=lfp_config_dict['dt'], fdst=lfp_config_dict['fraction'],
                                         maxEDist=lfp_config_dict['maxEDist'],
                                         seed=int(env.model_config['Random Seeds']['Local Field Potential']))
        if rank == 0:
            logger.info("*** LFP objects instantiated")
    lfp_time = time.time() - st

    st = time.time()
    if rank == 0:
        logger.info("*** Creating connections: time = %g seconds" % st)
    if env.cell_selection is None:
        connect_cells(env)
    else:
        connect_cell_selection(env)
    env.pc.set_maxstep(10.0)

    env.connectcellstime = time.time() - st

    if rank == 0:
        logger.info("*** Done creating connections: time = %g seconds" % time.time())
    if rank == 0:
        logger.info("*** Connections created in %g seconds" % env.connectcellstime)
    edge_count = int(sum([env.edge_count[dest] for dest in env.edge_count]))
    logger.info("*** Rank %i created %i connections" % (rank, edge_count))
    
    if env.profile_memory and rank == 0:
        profile_memory(logger)

    st = time.time()
    init_input_cells(env)
    env.mkstimtime = time.time() - st
    if rank == 0:
        logger.info("*** Stimuli created in %g seconds" % env.mkstimtime)
    setup_time = env.mkcellstime + env.mkstimtime + env.connectcellstime + env.connectgjstime + lfp_time
    max_setup_time = env.pc.allreduce(setup_time, 2)  ## maximum value
    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))
    tstop = (env.tstop + equilibration_duration)*float(env.n_trials)
    env.simtime = simtime.SimTimeEvent(env.pc, tstop, env.max_walltime_hours, env.results_write_time, max_setup_time)
    h.v_init = env.v_init
    h.stdinit()
    if env.optldbal or env.optlptbal:
        cx(env)
        ld_bal(env)
        if env.optlptbal:
            lpt_bal(env)


def run(env, output=True, shutdown=True, output_syn_spike_count=False):
    """
    Runs network simulation. Assumes that procedure `init` has been
    called with the network configuration provided by the `env`
    argument.

    :param env: an instance of the `dentate.Env` class
    :param output: if True, output spike and cell voltage trace data
    :param output_syn_spike_count: if True, output spike counts per pre-synaptic source for each gid
    """
    rank = int(env.pc.id())
    nhosts = int(env.pc.nhost())

    if output_syn_spike_count and env.cleanup:
        raise RuntimeError("Unable to compute synapse spike counts when cleanup is True")

    if rank == 0:
        if output:
            logger.info("Creating results file %s" % env.results_file_path)
            io_utils.mkout(env, env.results_file_path)

    if rank == 0:
        logger.info("*** Running simulation")

    rec_dt = None
    if env.recording_profile is not None:
        rec_dt = env.recording_profile.get('dt', None)
    if rec_dt is None:
        env.t_rec.record(h._ref_t)
    else:
        env.t_rec.record(h._ref_t, rec_dt)

    env.t_rec.resize(0)
    env.t_vec.resize(0)
    env.id_vec.resize(0)

    h.t = env.tstart
    env.simtime.reset()
    h.finitialize(env.v_init)
    h.finitialize(env.v_init)

    if rank == 0:
        logger.info("*** Completed finitialize")

    equilibration_duration = float(env.stimulus_config.get('Equilibration Duration', 0.))
    tstop = (env.tstop + equilibration_duration)*float(env.n_trials)

    if (env.checkpoint_interval is not None):
        if env.checkpoint_interval > 1.:
            tsegments = np.concatenate((np.arange(env.tstart, tstop, env.checkpoint_interval)[1:], np.asarray([tstop])))
        else:
            raise RuntimeError("Invalid checkpoint interval length")
    else:
        tsegments = np.asarray([tstop])

    for tstop_i in tsegments:
        if (h.t+env.dt) > env.simtime.tstop:
            break
        elif tstop_i < env.simtime.tstop:
            h.tstop = tstop_i
        else:
            h.tstop = env.simtime.tstop
        if rank == 0:
            logger.info("*** Running simulation up to %.2f ms" % h.tstop)
        env.pc.timeout(env.nrn_timeout)
        env.pc.psolve(h.tstop)
        if env.use_coreneuron:
            h.t = h.tstop
        if output:
            if rank == 0:
                logger.info("*** Writing spike data up to %.2f ms" % h.t)
            io_utils.spikeout(env, env.results_file_path, t_start=env.last_checkpoint, clear_data=env.checkpoint_clear_data)
            if env.recording_profile is not None:
                if rank == 0:
                    logger.info("*** Writing intracellular data up to %.2f ms" % h.t)
                io_utils.recsout(env, env.results_file_path, t_start=env.last_checkpoint, clear_data=env.checkpoint_clear_data)
            env.last_checkpoint = h.t
    if output_syn_spike_count:
       for pop_name in sorted(viewkeys(env.biophys_cells)):
           presyn_names = sorted(env.projection_dict[pop_name])
           synapses.write_syn_spike_count(env, pop_name, env.results_file_path,
                                          filters={'sources': presyn_names},
                                          write_kwds={'io_size': env.io_size})
            
    if rank == 0:
        logger.info("*** Simulation completed")

    if rank == 0 and output:
        io_utils.lfpout(env, env.results_file_path)
        
    if shutdown:
        del env.cells

    comptime = env.pc.step_time()
    cwtime = comptime + env.pc.step_wait()
    maxcw = env.pc.allreduce(cwtime, 2)
    meancomp = env.pc.allreduce(comptime, 1) / nhosts
    maxcomp = env.pc.allreduce(comptime, 2)

    gjtime = env.pc.vtransfer_time()

    gjvect = h.Vector()
    env.pc.allgather(gjtime, gjvect)
    meangj = gjvect.mean()
    maxgj = gjvect.max()

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
            logger.info("Load balance = %g" % (meancomp / maxcw))
        if meangj > 0:
            logger.info("Mean/max voltage transfer time: %g / %g seconds" % (meangj, maxgj))
            for i in range(nhosts):
                logger.debug("Voltage transfer time on host %i: %g seconds" % (i, gjvect.x[i]))

    if shutdown:
        env.pc.runworker()
        env.pc.done()
        h.quit()


