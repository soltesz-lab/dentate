
import sys, time, gc
import numpy as np
from mpi4py import MPI
from neuron import h
from neuroh5.io import NeuroH5TreeGen, population_ranges, append_cell_attributes
import click
from env import Env
import utils, synapses

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
def main(config, template_path, forest_path, populations, distribution, io_size, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    env = Env(comm=MPI.COMM_WORLD, configFile=config, templatePaths=template_path)
    h('objref nil, pc')
    h.load_file("nrngui.hoc")
    h.xopen("./lib.hoc")
    h.pc = h.ParallelContext()
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    (pop_ranges, _) = population_ranges(comm, forest_path)
    start_time = time.time()
    for population in populations:
        count = 0
        (population_start, _) = pop_ranges[population]
        template_name = env.celltypes[population]['templateName']
        h.find_template(h.pc, env.templatePaths, template_name)
        density_dict = env.celltypes[population]['synapses']['density']
        for gid, morph_dict in NeuroH5TreeGen(comm, forest_path, population, io_size=io_size):
            local_time = time.time()
            # mismatched_section_dict = {}
            synapse_dict = {}
            if gid is not None:
                print  'Rank %i gid: %i' % (rank, gid)
                cell = utils.new_cell(template_name, neurotree_dict=morph_dict, gid=gid)
                # this_mismatched_sections = cell.get_mismatched_neurotree_sections()
                # if this_mismatched_sections is not None:
                #    mismatched_section_dict[gid] = this_mismatched_sections
                cell_sec_dict = {'apical': cell.apical, 'basal': cell.basal, 'soma': cell.soma, 'ais': cell.ais}
                
                if distribution == 'uniform':
                    synapse_dict[gid-population_start] = synapses.distribute_uniform_synapses(gid, env.Synapse_Types, env.SWC_Types,
                                                                                              density_dict, morph_dict, cell_sec_dict)
                del cell
                print 'Rank %i took %i s to compute syn_locs for %s gid: %i' % (rank, time.time() - local_time, population, gid)
                count += 1
            else:
                print  'Rank %i gid is None' % rank
            # print 'Rank %i before append_cell_attributes' % rank
            append_cell_attributes(comm, forest_path, population, synapse_dict,
                                    namespace='Synapse Attributes', io_size=io_size, chunk_size=chunk_size,
                                    value_chunk_size=value_chunk_size)
            sys.stdout.flush()
            del synapse_dict
            gc.collect()
    # print 'Rank %i completed iterator' % rank

    # len_mismatched_section_dict_fragments = comm.gather(len(mismatched_section_dict), root=0)
        global_count = comm.gather(count, root=0)
        if rank == 0:
            print 'target: %s, %i ranks took %i s to compute syn_locs for %i cells' % (population, comm.size,
                                                                                        time.time() - start_time,
                                                                                        np.sum(global_count))
        # print '%i morphologies have mismatched section indexes' % np.sum(len_mismatched_section_dict_fragments)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("compute_synapse_locs.py") != -1,sys.argv)+1):])
