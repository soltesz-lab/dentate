
import sys, time, gc
import numpy as np
from mpi4py import MPI
from neuron import h
from neuroh5.io import NeuroH5TreeGen, read_population_ranges, append_cell_attributes
from env import Env
import utils, cells, synapses
import click

def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--populations", required=True, multiple=True, type=str)
@click.option("--distribution", type=str, default='uniform')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=10000)
def main(config, template_path, forest_path, populations, distribution, io_size, chunk_size, value_chunk_size, cache_size):

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
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    h.templatePaths = h.List()
    for path in env.templatePaths:
        h.templatePaths.append(h.Value(1,path))
    
    (pop_ranges, _) = read_population_ranges(comm, forest_path)
    start_time = time.time()
    for population in populations:
        print  'Rank %i population: %s' % (rank, population)
        count = 0
        (population_start, _) = pop_ranges[population]
        template_name = env.celltypes[population]['template']
        h.find_template(h.pc, h.templatePaths, template_name)
        template_class = eval('h.%s' % template_name)
        density_dict = env.celltypes[population]['synapses']['density']
        for gid, morph_dict in NeuroH5TreeGen(comm, forest_path, population, io_size=io_size):
            local_time = time.time()
            # mismatched_section_dict = {}
            synapse_dict = {}
            if gid is not None:
                print  'Rank %i gid: %i' % (rank, gid)
                cell = cells.make_neurotree_cell(template_class, neurotree_dict=morph_dict, gid=gid)
                # this_mismatched_sections = cell.get_mismatched_neurotree_sections()
                # if this_mismatched_sections is not None:
                #    mismatched_section_dict[gid] = this_mismatched_sections
                cell_sec_dict = {'apical': (cell.apical, None), 'basal': (cell.basal, None), 'soma': (cell.soma, None), 'axon': (cell.axon, 50.0)}
                cell_secidx_dict = {'apical': cell.apicalidx, 'basal': cell.basalidx, 'soma': cell.somaidx, 'axon': cell.axonidx}

                if distribution == 'uniform':
                    synapse_dict[gid-population_start] = synapses.distribute_uniform_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                              density_dict, morph_dict,
                                                                                              cell_sec_dict, cell_secidx_dict)
                elif distribution == 'poisson':
                    synapse_dict[gid-population_start] = synapses.distribute_poisson_synapses(gid, env.Synapse_Types, env.SWC_Types, env.layers,
                                                                                              density_dict, morph_dict,
                                                                                              cell_sec_dict, cell_secidx_dict)
                else:
                    raise Exception('Unknown distribution type: %s' % distribution)
                    
                del cell
                num_syns = len(synapse_dict[gid-population_start]['syn_ids'])
                print 'Rank %i took %i s to compute %d synapse locations for %s gid: %i' % (rank, time.time() - local_time, num_syns, population, gid)
                count += 1
            else:
                print  'Rank %i gid is None' % rank
            # print 'Rank %i before append_cell_attributes' % rank
            append_cell_attributes(comm, forest_path, population, synapse_dict,
                                    namespace='Synapse Attributes', io_size=io_size, chunk_size=chunk_size,
                                    value_chunk_size=value_chunk_size, cache_size=cache_size)
            sys.stdout.flush()
            del synapse_dict
            gc.collect()
    # print 'Rank %i completed iterator' % rank

    # len_mismatched_section_dict_fragments = comm.gather(len(mismatched_section_dict), root=0)
        global_count = comm.gather(count, root=0)
        if rank == 0:
            print 'target: %s, %i ranks took %i s to compute synapse locations for %i cells' % (population, comm.size,
                                                                                        time.time() - start_time,
                                                                                        np.sum(global_count))
        # print '%i morphologies have mismatched section indexes' % np.sum(len_mismatched_section_dict_fragments)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("distribute_synapse_locs.py") != -1,sys.argv)+1):])
