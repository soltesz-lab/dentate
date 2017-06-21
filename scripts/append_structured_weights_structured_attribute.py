from function_lib import *
from mpi4py import MPI
from itertools import izip
from neurotrees.io import NeurotreeAttrGen
from neurotrees.io import append_cell_attributes
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass

script_name = 'append_structured_weights_structured_attribute.py'


@click.command()
@click.option("--weights-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False), default=None)
@click.option("--weights-namespace", type=str, default='Weights')
@click.option("--structured-weights-namespace", type=str, default='Structured Weights')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--debug", is_flag=True)
def main(weights_path, weights_namespace, structured_weights_namespace, io_size, chunk_size, value_chunk_size,
         cache_size, debug):
    """

    :param weights_path:
    :param weights_namespace:
    :param structured_weights_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param debug:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print '%i ranks have been allocated' % comm.size
    sys.stdout.flush()

    population = 'GC'
    count = 0
    structured_count = 0
    start_time = time.time()
    weights_gen = NeurotreeAttrGen(MPI._addressof(comm), weights_path, population, io_size=io_size,
                                        cache_size=cache_size, namespace=weights_namespace)
    structured_weights_gen = NeurotreeAttrGen(MPI._addressof(comm), weights_path, population, io_size=io_size,
                                   cache_size=cache_size, namespace=structured_weights_namespace)
    if debug:
        attr_gen = ((weights_gen.next(), structured_weights_gen.next()) for i in xrange(10))
    else:
        attr_gen = izip(weights_gen, structured_weights_gen)
    for (gid, weights_dict), (structured_weights_gid, structured_weights_dict) in attr_gen:
        local_time = time.time()
        modified_dict = {}
        sorted_indexes = None
        sorted_weights = None
        sorted_structured_indexes = None
        sorted_structured_weights = None
        if gid is not None:
            if gid != structured_weights_gid:
                raise Exception('gid %i from weights_gen does not match gid %i from structured_weights_gen') % \
                      (gid, structured_weights_gid)
            sorted_indexes = weights_dict[weights_namespace]['syn_id'].argsort()
            sorted_weights = weights_dict[weights_namespace]['weight'][sorted_indexes]
            sorted_structured_indexes = structured_weights_dict[structured_weights_namespace]['syn_id'].argsort()
            sorted_structured_weights = \
                structured_weights_dict[structured_weights_namespace]['weight'][sorted_structured_indexes]
            if not np.all(weights_dict[weights_namespace]['syn_id'][sorted_indexes] ==
                          structured_weights_dict[structured_weights_namespace]['syn_id'][sorted_structured_indexes]):
                raise Exception('gid %i: sorted syn_ids from weights_namespace do not match '
                                'structured_weights_namespace') % gid
            modify_weights = not np.all(sorted_weights == sorted_structured_weights)
            modified_dict[gid] = {'structured': np.array([int(modify_weights)], dtype='uint32')}
            print 'Rank %i: %s gid %i took %.2f s to check for structured weights: %s' % \
                  (rank, population, gid, time.time() - local_time, str(modify_weights))
            if modify_weights:
                structured_count += 1
            count += 1
        if not debug:
            append_cell_attributes(MPI._addressof(comm), weights_path, population, modified_dict,
                                   namespace=structured_weights_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        else:
            comm.barrier()
        del sorted_indexes
        del sorted_weights
        del sorted_structured_indexes
        del sorted_structured_weights
        del modified_dict
        gc.collect()
        sys.stdout.flush()

    global_count = comm.gather(count, root=0)
    global_structured_count = comm.gather(structured_count, root=0)
    if rank == 0:
        print '%i ranks processed %i %s cells (%i assigned structured weights) in %.2f s' % \
              (comm.size, np.sum(global_count), population, np.sum(global_structured_count),
               time.time() - start_time)


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1, sys.argv) + 1):])
