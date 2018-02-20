
from dentate.utils import *
import numpy as np
from mpi4py import MPI
from neuron import h
from neuroh5.io import NeuroH5TreeGen, read_population_ranges, append_cell_attributes
import dentate
from dentate.env import Env
import dentate.cells as cells
import dentate.synapses as synapses
import click
import logging

script_name="distribute_synapse_locs.py"
logger = logging.getLogger(script_name)


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--distribution", type=str, default='uniform')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=10000)
@click.option("--verbose", "-v", is_flag=True)
def main(config, template_path, forest_path, populations, distribution, io_size, chunk_size, value_chunk_size,
         cache_size, verbose):
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

    if verbose:
        logger.setLevel(logging.INFO)
        
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
    sys.stdout.flush()

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePaths.append(h.Value(1,path))
    
    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()
    for population in populations:
        logger.info('Rank %i population: %s' % (rank, population))
        count = 0
        (population_start, _) = pop_ranges[population]
        template_name = env.celltypes[population]['template']
        h.find_template(h.pc, h.templatePaths, template_name)
        template_class = eval('h.%s' % template_name)
        density_dict = env.celltypes[population]['synapses']['density']
        for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, comm=comm, topology=True):
            local_time = time.time()
            synapse_dict = {}
            if gid is not None:
                logger.info('Rank %i gid: %i' % (rank, gid))
                cell = cells.make_neurotree_cell(template_class, neurotree_dict=morph_dict, gid=gid)
                cell_sec_dict = {'apical': (cell.apical, None), 'basal': (cell.basal, None), 'soma': (cell.soma, None), 'ais': (cell.ais, None)}
                cell_secidx_dict = {'apical': cell.apicalidx, 'basal': cell.basalidx, 'soma': cell.somaidx, 'ais': cell.aisidx}

                if distribution == 'uniform':
                    synapse_dict[gid] = synapses.distribute_uniform_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                              density_dict, morph_dict,
                                                                                              cell_sec_dict, cell_secidx_dict)
                elif distribution == 'poisson':
                    if rank == 0:
                        verbose_flag = verbose
                    else:
                        verbose_flag = False
                    synapse_dict[gid] = synapses.distribute_poisson_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                              density_dict, morph_dict,
                                                                                              cell_sec_dict, cell_secidx_dict, verbose=verbose_flag)
                else:
                    raise Exception('Unknown distribution type: %s' % distribution)
                    
                del cell
                num_syns = len(synapse_dict[gid]['syn_ids'])
                logger.info('Rank %i took %i s to compute %d synapse locations for %s gid: %i' % (rank, time.time() - local_time, num_syns, population, gid))
                count += 1
            else:
                logger.info('Rank %i gid is None' % rank)
            append_cell_attributes(forest_path, population, synapse_dict,
                                    namespace='Synapse Attributes', comm=comm, io_size=io_size, chunk_size=chunk_size,
                                    value_chunk_size=value_chunk_size, cache_size=cache_size)
            sys.stdout.flush()
            del synapse_dict
            gc.collect()

        global_count = comm.gather(count, root=0)
        if rank == 0:
            logger.info('target: %s, %i ranks took %i s to compute synapse locations for %i cells' % (population, comm.size,time.time() - start_time,np.sum(global_count)))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
