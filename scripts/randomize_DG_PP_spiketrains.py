
import sys, time, gc
import numpy as np
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, read_population_ranges
import click
import utils

try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

script_name = 'randomize_DG_PP_spiketrains.py'


@click.command()
@click.option("--stimulus-path", '-p', required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-stimulus-namespace", '-i', type=str, default='Vector Stimulus')
@click.option("--output-stimulus-namespace", '-o', type=str, default='Randomized Vector Stimulus')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--seed-offset", type=int, default=9)
@click.option("--debug", is_flag=True)
def main(stimulus_path, input_stimulus_namespace, output_stimulus_namespace, io_size, chunk_size, value_chunk_size, cache_size, 
         seed_offset, debug):
    """
    :param input_stimulus_namespace: str
    :param output_stimulus_namespace: str
    :param io_size: int
    :param chunk_size: int
    :param value_chunk_size: int
    :param cache_size: int
    :param seed_offset: int
    :param debug: bool
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    seed_offset *= 2e6
    np.random.seed(int(seed_offset))
                

    population_ranges = read_population_ranges(comm, stimulus_path)[0]

    for population in ['LPP']:
        population_start = population_ranges[population][0]
        population_count = population_ranges[population][1]

        if rank == 0:
            random_gids = np.random.randint(0, high=population_count-1, size=population_count)
        else:
            random_gids = None
        random_gids = comm.bcast(random_gids, root=0)

        count = 0
        start_time = time.time()

        attr_gen = NeuroH5CellAttrGen(comm, stimulus_path, population, io_size=io_size,
                                      cache_size=cache_size, namespace=input_stimulus_namespace)
        if debug:
            attr_gen_wrapper = (attr_gen.next() for i in xrange(2))
        else:
            attr_gen_wrapper = attr_gen
        for gid, stimulus_dict in attr_gen_wrapper:
            local_time = time.time()
            new_response_dict = {}
            if gid is not None:

                random_gid = random_gids[gid-population_start]
                new_response_dict[random_gid] = {'rate': stimulus_dict['rate'],
                                                 'spiketrain': np.asarray(stimulus_dict['spiketrain'], dtype=np.float32),
                                                 'modulation': stimulus_dict['modulation'],
                                                 'peak_index': stimulus_dict['peak_index'] }

                print 'Rank %i; source: %s; assigned spike trains for gid %i to gid %i' % \
                      (rank, population, gid, random_gid+population_start)
                count += 1
            if not debug:
                append_cell_attributes(comm, stimulus_path, population, new_response_dict,
                                       namespace=output_stimulus_namespace,
                                       io_size=io_size, chunk_size=chunk_size,
                                       value_chunk_size=value_chunk_size)
            sys.stdout.flush()
            del new_response_dict
            gc.collect()

        global_count = comm.gather(count, root=0)
        if rank == 0:
            print '%i ranks randomized spike trains for %i cells in %.2f s' % (comm.size, np.sum(global_count),
                                                                               time.time() - start_time)


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
