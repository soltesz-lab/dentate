import sys, os, gc, string, time, click, logging, itertools
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from neuron import h
from neuroh5.io import NeuroH5TreeGen, read_population_ranges, append_cell_attributes
import h5py
import dentate
from dentate.env import Env
from dentate import cells, synapses, utils
from dentate.utils import list_find

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook



def update_syn_stats(env, syn_stats_dict, syn_dict):

    syn_type_excitatory = env.Synapse_Types['excitatory']
    syn_type_inhibitory = env.Synapse_Types['inhibitory']

    this_syn_stats_dict = { 'section': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                            'layer': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                            'swc_type': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                            'total': { 'excitatory': 0, 'inhibitory': 0 } }

    for (syn_id,syn_sec,syn_type,swc_type,syn_layer) in \
        itertools.izip(syn_dict['syn_ids'],
                       syn_dict['syn_secs'],
                       syn_dict['syn_types'],
                       syn_dict['swc_types'],
                       syn_dict['syn_layers']):
        
        if syn_type == syn_type_excitatory:
            syn_type_str = 'excitatory'
        elif syn_type == syn_type_inhibitory:
            syn_type_str = 'inhibitory'
        else:
            raise ValueError('Unknown synapse type %s' % str(syn_type))

        syn_stats_dict['section'][syn_sec][syn_type_str] += 1
        syn_stats_dict['layer'][syn_layer][syn_type_str] += 1
        syn_stats_dict['swc_type'][swc_type][syn_type_str] += 1
        syn_stats_dict['total'][syn_type_str] += 1

        this_syn_stats_dict['section'][syn_sec][syn_type_str] += 1
        this_syn_stats_dict['layer'][syn_layer][syn_type_str] += 1
        this_syn_stats_dict['swc_type'][swc_type][syn_type_str] += 1
        this_syn_stats_dict['total'][syn_type_str] += 1

    return this_syn_stats_dict


def global_syn_summary(comm, syn_stats, global_count, root):
    res = []
    for population in syn_stats.keys():
        pop_syn_stats = syn_stats[population]
        for part in ['layer', 'swc_type']:
            syn_stats_dict = pop_syn_stats[part]
            for part_name in syn_stats_dict.keys():
                for syn_type in syn_stats_dict[part_name].keys():
                    global_syn_count = comm.gather(syn_stats_dict[part_name][syn_type], root=root)
                    if comm.rank == root:
                        res.append("%s %s %s: mean %s synapses per cell: %f" % (population, part, part_name, syn_type, np.sum(global_syn_count)/global_count))
        total_syn_stats_dict = pop_syn_stats['total']
        for syn_type in total_syn_stats_dict.keys():
            global_syn_count = comm.gather(total_syn_stats_dict[syn_type], root=root)
            if comm.rank == root:
                res.append("%s: mean %s synapses per cell: %f" % (population, syn_type, np.sum(global_syn_count)/global_count))
        
    return string.join(res, '\n')

def local_syn_summary(syn_stats_dict):
    res = []
    for part_name in ['layer','swc_type']:
        for part_type in syn_stats_dict[part_name].keys():
            syn_count_dict = syn_stats_dict[part_name][part_type]
            for syn_type, syn_count in syn_count_dict.iteritems():
                res.append("%s %i: %s synapses: %i" % (part_name, part_type, syn_type, syn_count))
    return string.join(res, '\n')


def check_syns(gid, morph_dict, syn_stats_dict, seg_density_per_sec, layer_set_dict, swc_set_dict, env, logger):

    layer_stats = syn_stats_dict['layer']
    swc_stats = syn_stats_dict['swc_type']

    warning_flag = False
    for syn_type, layer_set in layer_set_dict.iteritems():
        for layer in layer_set:
            if layer_stats.has_key(layer):
                if layer_stats[layer][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning('Rank %d: incomplete synapse layer set for cell %d: %s' % (env.comm.Get_rank(), gid, str(dict(layer_stats.iteritems()))))
        logger.info('layer_set_dict: %s' % (str(dict(layer_set_dict.iteritems()))))
        logger.info('gid %d: seg_density_per_sec: %s' % (gid, str(seg_density_per_sec)))
        logger.info('gid %d: morph_dict: %s' % (gid, str(morph_dict)))
    for syn_type, swc_set in swc_set_dict.iteritems():
        for swc_type in swc_set:
            if swc_stats.has_key(swc_type):
                if swc_stats[swc_type][syn_type] <= 0:
                    warning_flag = True
            else:
                warning_flag = True
    if warning_flag:
        logger.warning('Rank %d: incomplete synapse swc type set for cell %d: %s' % (env.comm.Get_rank(), gid, str(dict(swc_stats.iteritems()))))
        logger.info('swc_set_dict: %s' % (str(dict(swc_set_dict.iteritems()))))
        logger.info('gid %d: seg_density_per_sec: %s' % (gid, str(seg_density_per_sec)))
        logger.info('gid %d: morph_dict: %s' % (gid, str(morph_dict)))
                

            
        
@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--output-path", type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--distribution", type=str, default='uniform')
@click.option("--traversal-order", type=str, default='bfs')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, template_path, output_path, forest_path, populations, distribution, traversal_order, io_size, chunk_size, value_chunk_size,
         cache_size, verbose, dry_run):
    """

    :param config:
    :param template_path:
    :param forest_path:
    :param populations:
    :param distribution:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    env = Env(comm=MPI.COMM_WORLD, configFile=config, templatePaths=template_path)
    h('objref nil, pc, templatePaths')
    h.load_file("nrngui.hoc")
    h.load_file("./templates/Value.hoc")
    h.xopen("./lib.hoc")
    h.pc = h.ParallelContext()
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePaths.append(h.Value(1,path))

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
        comm.barrier()
        
    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()
    syn_stats = {}
    for population in populations:
        logger.info('Rank %i population: %s' % (rank, population))
        count = 0
        (population_start, _) = pop_ranges[population]
        template_name = env.celltypes[population]['template']
        h.find_template(h.pc, h.templatePaths, template_name)
        template_class = getattr(h, env.celltypes[population]['template'])
        density_dict = env.celltypes[population]['synapses']['density']
        layer_set_dict = defaultdict(set)
        swc_set_dict = defaultdict(set)
        for sec_name, sec_dict in density_dict.iteritems():
            for syn_type, syn_dict in sec_dict.iteritems():
                swc_set_dict[syn_type].add(env.SWC_Types[sec_name])
                for layer_name in syn_dict.keys():
                    if layer_name != 'default':
                        layer = env.layers[layer_name]
                        layer_set_dict[syn_type].add(layer)
        
        syn_stats_dict = { 'section': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'layer': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'swc_type': defaultdict(lambda: { 'excitatory': 0, 'inhibitory': 0 }), \
                           'total': { 'excitatory': 0, 'inhibitory': 0 } }
                           
        for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=True):
            local_time = time.time()
            synapse_dict = {}
            if gid is not None:
                logger.info('Rank %i gid: %i' % (rank, gid))
                cell = cells.make_neurotree_cell(template_class, neurotree_dict=morph_dict, gid=gid)
                cell_sec_dict = {'apical': (cell.apical, None), 'basal': (cell.basal, None), 'soma': (cell.soma, None), 'ais': (cell.ais, None)}
                cell_secidx_dict = {'apical': cell.apicalidx, 'basal': cell.basalidx, 'soma': cell.somaidx, 'ais': cell.aisidx}

                if distribution == 'uniform':
                    syn_dict, seg_density_per_sec = synapses.distribute_uniform_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                    density_dict, morph_dict,
                                                                    cell_sec_dict, cell_secidx_dict,
                                                                    traversal_order=traversal_order)
                    
                elif distribution == 'poisson':
                    syn_dict, seg_density_per_sec = synapses.distribute_poisson_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                    density_dict, morph_dict,
                                                                    cell_sec_dict, cell_secidx_dict,
                                                                    traversal_order=traversal_order)
                else:
                    raise Exception('Unknown distribution type: %s' % distribution)

                synapse_dict[gid] = syn_dict
                this_syn_stats = update_syn_stats (env, syn_stats_dict, syn_dict)
                check_syns(gid, morph_dict, this_syn_stats, seg_density_per_sec, layer_set_dict, swc_set_dict, env, logger)
                
                del cell
                num_syns = len(synapse_dict[gid]['syn_ids'])
                logger.info('Rank %i took %i s to compute %d synapse locations for %s gid: %i' % (rank, time.time() - local_time, num_syns, population, gid))
                logger.info('%s gid %i synapses: %s' % (population, gid, local_syn_summary(this_syn_stats)))
                count += 1
            else:
                logger.info('Rank %i gid is None' % rank)
            if not dry_run:
                append_cell_attributes(output_path, population, synapse_dict,
                                       namespace='Synapse Attributes', comm=comm, io_size=io_size, chunk_size=chunk_size,
                                       value_chunk_size=value_chunk_size, cache_size=cache_size)
            syn_stats[population] = syn_stats_dict
            del synapse_dict
            gc.collect()

        global_count = comm.gather(count, root=0)
        summary = global_syn_summary(comm, syn_stats, np.sum(global_count), root=0)
        if rank == 0:
            logger.info('target: %s, %i ranks took %i s to compute synapse locations for %i cells' % (population, comm.size,time.time() - start_time,np.sum(global_count)))
            logger.info(summary)
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

