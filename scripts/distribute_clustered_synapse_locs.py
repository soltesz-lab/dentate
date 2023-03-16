import os, sys, gc, logging, string, time, math, itertools, pprint
from mpi4py import MPI
import click
from collections import defaultdict
import numpy as np
import dentate
from dentate import cells, neuron_utils, synapses, utils
from dentate.env import Env
from dentate.neuron_utils import configure_hoc_env
from dentate.cells import load_cell_template
from dentate import minmax_kmeans
from neuroh5.io import NeuroH5CellAttrGen, scatter_read_trees, scatter_read_cell_attributes, append_cell_attributes, read_population_ranges
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    sys.stdout.flush()
    sys.stderr.flush()
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

def merge_dict(d1, d2, datatype):
    dd = {}
    for d in (d1, d2):
        dd.update(d)
    return dd

mpi_op_merge_dict = MPI.Op.Create(merge_dict, commute=True)
            
        
@click.command()
@click.option("--config", required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True), default='config')
@click.option("--template-path", type=str)
@click.option("--output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--structured-weights-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--synapse-clusters-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--arena-id", type=str)
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--write-size", type=int, default=0)
@click.option("--attr-gen-cache-size", type=int, default=10)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
@click.option("--debug", is_flag=True)
def main(config, config_prefix, template_path, output_path, forest_path, structured_weights_path, synapse_clusters_path, populations, arena_id, io_size, chunk_size, value_chunk_size,
         write_size, attr_gen_cache_size, verbose, dry_run, debug):
    """

    :param config:
    :param config_prefix:
    :param template_path:
    :param forest_path:
    :param populations:
    :param distribution:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if structured_weights_path is None:
        structured_weights_path = forest_path
        
    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, template_paths=template_path)

    configure_hoc_env(env)
    
    if io_size == -1:
        io_size = env.comm.size

    if output_path is None:
        output_path = forest_path

    if not dry_run:
        if rank==0:
            if not os.path.isfile(output_path):
                input_file  = h5py.File(forest_path,'r')
                output_file = h5py.File(output_path,'w')
                input_file.copy('/H5Types',output_file)
                input_file.close()
                output_file.close()
        env.comm.barrier()

    input_rank_namespace = f"Input Rank Structured Weights {arena_id}"
    syn_clusters_namespace = f"Synapse Clusters {arena_id}"
    
    (pop_ranges, _) = read_population_ranges(forest_path, comm=env.comm)

    for population in populations:
        start_time = time.time()
        
        logger.info('Rank %i population: %s' % (rank, population))
        (population_start, _) = pop_ranges[population]
        template_class = load_cell_template(env, population, bcast_template=True)

        projection_config = env.connection_config[population]
        projection_synapse_dict = {env.Populations[source_population]:
                                   (projection_config[source_population].type,
                                    set(projection_config[source_population].layers),
                                    set(projection_config[source_population].sections))
                                   for source_population in projection_config}
        
        density_config_dict = env.celltypes[population]['synapses']['density']

        cell_dicts = {}

        node_allocation = set([])

        trees, _ = scatter_read_trees(forest_path, population, topology=True, comm=env.comm, io_size=io_size)
        for this_gid, this_morph_dict in trees:
            morph_dict = this_morph_dict
            cell = cells.make_neurotree_hoc_cell(template_class, neurotree_dict=morph_dict, gid=this_gid)

            cell_sec_dict = {'apical': (cell.apical, None), 
                             'basal': (cell.basal, None), 
                             'soma': (cell.soma, None),
                             'ais': (cell.ais, None), 
                             'hillock': (cell.hillock, None)}
            cell_secidx_dict = {'apical': cell.apicalidx, 
                                'basal': cell.basalidx, 
                                'soma': cell.somaidx, 
                                'ais': cell.aisidx, 
                                'hillock': cell.hilidx}
            cell_dicts[this_gid] = { 'cell': cell,
                                     'morph_dict': morph_dict,
                                     'sec_dict': cell_sec_dict, 
                                     'secidx_dict': cell_secidx_dict}
            node_allocation.add(this_gid)

        env.node_allocation = node_allocation

        env.comm.barrier()


        syn_ids_ind = None
        syn_locs_ind = None
        syn_layers_ind = None
        syn_types_ind = None
        swc_types_ind = None
        syn_secs_ind = None
        syn_locs_ind = None

        synapses_attr_gen = NeuroH5CellAttrGen(forest_path, population,
                                               namespace="Synapse Attributes",
                                               mask=set(['syn_ids', 'syn_locs', 'syn_secs', 
                                                         'syn_layers', 'syn_types', 'swc_types']), 
                                               return_type='tuple',
                                               comm=env.comm, io_size=io_size,
                                               cache_size=attr_gen_cache_size,
                                               node_allocation=env.node_allocation)

        for this_gid, this_syn_attrs in synapses_attr_gen:

            if this_gid is not None:
                (attr_tuple, attr_tuple_index) = this_syn_attrs
                if syn_ids_ind is None:
                    syn_ids_ind = attr_tuple_index.get('syn_ids', None)
                    syn_locs_ind = attr_tuple_index.get('syn_locs', None)
                    syn_layers_ind = attr_tuple_index.get('syn_layers', None)
                    syn_types_ind = attr_tuple_index.get('syn_types', None)
                    swc_types_ind = attr_tuple_index.get('swc_types', None)
                    syn_secs_ind = attr_tuple_index.get('syn_secs', None)
                    syn_locs_ind = attr_tuple_index.get('syn_locs', None)
                    
                cell_dicts[this_gid]['syn_attrs'] = attr_tuple

        env.comm.barrier()
        if rank == 0:
            logger.info("done reading synapse attributes")

        gids = []
        cell_syn_clusters = {}

        input_rank_attr_dict = scatter_read_cell_attributes(structured_weights_path, population,
                                                            namespaces=[input_rank_namespace],
                                                            return_type='tuple',
                                                            comm=env.comm, io_size=io_size,
                                                            node_allocation=env.node_allocation)
        (syn_rank_attr_iter, syn_rank_attr_tuple_index) = input_rank_attr_dict[input_rank_namespace]
        syn_id_rank_ind = syn_rank_attr_tuple_index.get('syn_id', None)
        syn_source_rank_ind = syn_rank_attr_tuple_index.get('source', None)
        syn_rank_ind = syn_rank_attr_tuple_index.get('rank', None)
        
        for this_gid, this_syn_rank_attr_tuple in syn_rank_attr_iter:
            
            syn_attrs = cell_dicts[this_gid]['syn_attrs']
            syn_source_rank_dict = { syn_id: (rank, source) for syn_id, source, rank in
                                     zip(this_syn_rank_attr_tuple[syn_id_rank_ind], 
                                         this_syn_rank_attr_tuple[syn_source_rank_ind], 
                                         this_syn_rank_attr_tuple[syn_rank_ind]) }
            syn_attrs_dict = { syn_id: (syn_source_rank_dict[syn_id][0], 
                                        syn_source_rank_dict[syn_id][1], 
                                        syn_type, swc_type, layer, sec) 
                               for syn_id, syn_type, layer, swc_type, sec in 
                               zip(syn_attrs[syn_ids_ind], 
                                   syn_attrs[syn_types_ind],
                                   syn_attrs[syn_layers_ind],
                                   syn_attrs[swc_types_ind],
                                   syn_attrs[syn_secs_ind]) if syn_id in syn_source_rank_dict }    
            cell_dicts[this_gid]['syn_attrs'] = syn_attrs_dict
            gids.append(this_gid)

        num_gids = len(gids)
        max_n_gids = env.comm.allreduce(num_gids, op=MPI.MAX)

        env.comm.barrier()
        
        if synapse_clusters_path is not None:
            syn_clusters_attr_dict = scatter_read_cell_attributes(synapse_clusters_path, population,
                                                                namespaces=[syn_clusters_namespace],
                                                                return_type='tuple',
                                                                comm=env.comm, io_size=io_size,
                                                                node_allocation=env.node_allocation)
            (syn_clusters_attr_iter, syn_clusters_attr_tuple_index) = syn_clusters_attr_dict[syn_clusters_namespace]
            syn_id_ind = syn_clusters_attr_tuple_index.get('syn_id', None)
            syn_cluster_id_ind = syn_clusters_attr_tuple_index.get('cluster_id', None)
            for this_gid, this_syn_cluster_attr_tuple in syn_clusters_attr_iter:
                syn_ids = this_syn_cluster_attr_tuple[syn_id_ind]
                syn_clusters = this_syn_cluster_attr_tuple[syn_cluster_id_ind]
                cell_syn_clusters[this_gid] = list(zip(syn_ids, syn_clusters))

        else:

            for i in range(max_n_gids):

                this_gid = None
                if i < num_gids:
                    this_gid = gids[i]

                if this_gid is None:
                    continue

                if rank == 0:
                    logger.info(f'Creating synapse clusters for gid {this_gid}...')
                local_time = time.time()
                cell_secidx_dict = cell_dicts[this_gid]['secidx_dict']
                syn_attrs_dict = cell_dicts[this_gid]['syn_attrs']
                syn_ids = list(syn_attrs_dict.keys())
                num_syns = len(syn_ids)
                syn_secs_array = np.fromiter([syn_attrs_dict[syn_id][5] for syn_id in syn_ids], dtype=int)
                syn_ranks_array = np.fromiter([syn_attrs_dict[syn_id][0] for syn_id in syn_ids], dtype=np.float32).reshape((-1,1))
                syn_sec_ids, syn_sec_counts = np.unique(syn_secs_array, return_counts=True)
                mean_syn_sec_count = np.mean(syn_sec_counts)
                cluster_max_size=mean_syn_sec_count
                k = int(len(cell_secidx_dict['apical'].as_numpy()))
                clusters, centers = minmax_kmeans.minsize_kmeans(syn_ranks_array, k, 1, max_size=cluster_max_size, verbose=verbose)
                if clusters is None:
                    continue

                if rank == 0:
                    logger.info(f"Rank {rank}: synapse clusters for gid {this_gid}: {np.unique(clusters, return_counts=True)}; "
                                f"cluster centers: {np.sort(np.concatenate(centers))}")
                    logger.info(f'Rank {rank} took {time.time() - local_time:.01f} s to compute clustering for '
                                f'{num_syns} synapse locations for {population} gid {this_gid}')
                cell_syn_clusters[this_gid] = list(zip(syn_ids, clusters))

                if debug and i >= 2:
                    break

        env.comm.barrier()

        gid_count = 0
        gid_synapse_dict = {}
        gid_cluster_dict = {}
        for i in range(max_n_gids):

            this_gid = None
            if i < num_gids:
                this_gid = gids[i]

            if this_gid is not None:

                logger.info(f'Rank {rank}: distributing clustered synapses for gid {this_gid}... ')
            
                local_time = time.time()

                random_seed = env.model_config['Random Seeds']['Synapse Locations'] + this_gid

                
                syn_attrs_dict = cell_dicts[this_gid]['syn_attrs']

                if debug:
                    syn_secs = []
                    for syn_id in syn_attrs_dict:
                        (_, _, syn_type, swc_type, layer, syn_sec) = syn_attrs_dict[syn_id]
                        syn_secs.append(syn_sec)
                        
                    logger.info(f"Unclustered synapse sections: {pprint.pformat(np.unique(syn_secs, return_counts=True))}")
                
                
                if this_gid not in cell_syn_clusters:
                    continue

                syn_clusters = cell_syn_clusters[this_gid]
                syn_cluster_dict = {syn_id: cluster_id for syn_id, cluster_id in syn_clusters}
                syn_cluster_attrs_dict = defaultdict(lambda: defaultdict(list))
                num_syns = len(syn_cluster_dict)
                # Separate out clusters into syn_type, swc_type, layer
                for syn_id, cluster_id in syn_cluster_dict.items():
                    (_, _, syn_type, swc_type, layer, _) = syn_attrs_dict[syn_id]
                    syn_cluster_attrs_dict[(syn_type, swc_type, layer)][cluster_id].append(syn_id)

                cell_sec_dict = cell_dicts[this_gid]['sec_dict']
                cell_secidx_dict = cell_dicts[this_gid]['secidx_dict']
                cell_morph_dict = cell_dicts[this_gid]['morph_dict']
            
                syn_dict, seg_density_per_sec = synapses.distribute_clustered_poisson_synapses(random_seed, env.Synapse_Types,
                                                                                               env.SWC_Types, env.layers,
                                                                                               density_config_dict, cell_morph_dict,
                                                                                               cell_sec_dict, cell_secidx_dict,
                                                                                               syn_cluster_attrs_dict)

                assert(len(syn_dict['syn_ids']) == num_syns)
                if debug:
                    logger.info(f"Clustered synapse sections: {pprint.pformat(np.unique(syn_dict['syn_secs'], return_counts=True))}")
                gid_synapse_dict[this_gid] = syn_dict
                syn_ids = []
                cluster_ids = []
                for syn_id, cluster_id in syn_clusters:
                    syn_ids.append(syn_id)
                    cluster_ids.append(cluster_id)
                gid_cluster_dict[this_gid] = { 'syn_id': np.asarray(syn_ids, dtype=np.uint32),
                                               'cluster_id': np.asarray(cluster_ids, dtype=np.uint16) }
                
                logger.info(f'Rank {rank} took {time.time() - local_time:.01f} s to compute {num_syns} '
                            f'clustered synapse locations for {population} gid: {this_gid}')

                gid_count += 1
            
            if (not dry_run) and (write_size > 0) and (i % write_size == 0):
                if synapse_clusters_path != output_path:
                    append_cell_attributes(output_path, population, gid_cluster_dict,
                                           namespace=syn_clusters_namespace,
                                           comm=env.comm, io_size=io_size, 
                                           chunk_size=chunk_size, 
                                           value_chunk_size=value_chunk_size)
                append_cell_attributes(output_path, population, gid_synapse_dict,
                                       namespace=f'Clustered Synapse Attributes {arena_id}', 
                                       comm=env.comm, io_size=io_size, 
                                       chunk_size=chunk_size, 
                                       value_chunk_size=value_chunk_size)
                gid_synapse_dict = {}
                gid_cluster_dict = {}

            if debug and i == 2:
                break


        env.comm.barrier()
        if not dry_run:
            if synapse_clusters_path != output_path:
                append_cell_attributes(output_path, population, gid_cluster_dict,
                                       namespace=syn_clusters_namespace,
                                       comm=env.comm, io_size=io_size, 
                                       chunk_size=chunk_size, 
                                       value_chunk_size=value_chunk_size)
            append_cell_attributes(output_path, population, gid_synapse_dict,
                                   namespace=f'Clustered Synapse Attributes {arena_id}', 
                                   comm=env.comm, io_size=io_size, 
                                   chunk_size=chunk_size, 
                                   value_chunk_size=value_chunk_size)

        global_count = env.comm.reduce(gid_count, op=MPI.SUM, root=0)
        if rank == 0:
            logger.info(f"target: {population}, {env.comm.size} ranks took {time.time() - start_time:.01f} s "
                        f"to compute clustered synapse locations for {global_count} cells")

        env.comm.barrier()
            
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
