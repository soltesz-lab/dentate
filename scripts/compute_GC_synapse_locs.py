from specify_cells import *
from mpi4py import MPI
from neurotrees.io import NeurotreeGen
from neurotrees.io import append_cell_attributes
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


def list_find (f, lst):
    i=0
    for x in lst:
        if f(x):
            return i
        else:
            i=i+1
    return None

@click.command()
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
def main(forest_path, io_size, chunk_size, value_chunk_size):

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    population = 'GC'
    count = 0
    start_time = time.time()
    for gid, morph_dict in NeurotreeGen(MPI._addressof(comm), forest_path, population, io_size=io_size):
        local_time = time.time()
        # mismatched_section_dict = {}
        synapse_dict = {}
        if gid is not None:
            print('Rank %i gid: %i' % (rank, gid))
            cell = DG_GC(neurotree_dict=morph_dict, gid=gid, full_spines=False)
            # this_mismatched_sections = cell.get_mismatched_neurotree_sections()
            # if this_mismatched_sections is not None:
            #    mismatched_section_dict[gid] = this_mismatched_sections
            synapse_dict[gid] = cell.export_neurotree_synapse_attributes()
            del cell
            print('Rank %i took %i s to compute syn_locs for %s gid: %i' % (rank, time.time() - local_time, population, gid))
            count += 1
        else:
            print('Rank %i gid is None' % rank)
        # print 'Rank %i before append_cell_attributes' % rank
        append_cell_attributes(MPI._addressof(comm), forest_path, population, synapse_dict,
                                namespace='Synapse_Attributes', io_size=io_size, chunk_size=chunk_size,
                                value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del synapse_dict
        gc.collect()
    # print 'Rank %i completed iterator' % rank

    # len_mismatched_section_dict_fragments = comm.gather(len(mismatched_section_dict), root=0)
    global_count = comm.gather(count, root=0)
    if rank == 0:
        print('target: %s, %i ranks took %i s to compute syn_locs for %i cells' % (population, comm.size,
                                                                                       time.time() - start_time,
                                                                                       np.sum(global_count)))
        # print '%i morphologies have mismatched section indexes' % np.sum(len_mismatched_section_dict_fragments)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find("compute_GC_synapse_locs.py") != -1,sys.argv)+1):])
