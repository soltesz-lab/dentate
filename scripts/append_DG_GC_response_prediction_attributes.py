from function_lib import *
from mpi4py import MPI
from neuroh5.io import NeuroH5CellAttrGen, append_cell_attributes, bcast_cell_attributes, population_ranges
import click


try:
    import mkl
    mkl.set_num_threads(1)
except:
    pass


script_name = 'append_DG_GC_response_prediction_attributes.py'

example_features_path = '../morphologies/dentate_Full_Scale_Control_selectivity_20170615.h5'
example_connectivity_path = '../morphologies/DGC_forest_connectivity_20170427.h5'


@click.command()
@click.option("--features-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--prediction-namespace", type=str, default='Response Prediction')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--trajectory-id", type=int, default=0)
@click.option("--debug", is_flag=True)
def main(features_path, prediction_namespace, io_size, chunk_size, value_chunk_size, cache_size,
         trajectory_id, debug):
    """

    :param features_path:
    :param prediction_namespace:
    :param io_size:
    :param chunk_size:
    :param value_chunk_size:
    :param cache_size:
    :param trajectory_id:
    :param debug:
    """
    comm = MPI.COMM_WORLD
    rank = comm.rank

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        print('%i ranks have been allocated' % comm.size)
    sys.stdout.flush()

    prediction_namespace = prediction_namespace+' '+str(trajectory_id)

    target_population = 'GC'
    count = 0
    start_time = time.time()
    attr_gen = NeuroH5CellAttrGen(comm, features_path, target_population, io_size=io_size,
                                cache_size=cache_size, namespace=prediction_namespace)
    if debug:
        attr_gen_wrapper = (next(attr_gen) for i in range(2))
    else:
        attr_gen_wrapper = attr_gen
    for gid, response_dict in attr_gen_wrapper:
        local_time = time.time()
        response_attr_dict = {}
        response = None
        if gid is not None:
            response_attr_dict[gid] = {}
            response = response_dict[prediction_namespace]['waveform']
            baseline = np.mean(response[np.where(response <= np.percentile(response, 10.))[0]])
            peak = np.mean(response[np.where(response >= np.percentile(response, 90.))[0]])
            modulation = peak / baseline - 1.
            peak_index = np.where(response == np.max(response))[0][0]
            response_attr_dict[gid]['modulation'] = np.array([modulation], dtype='float32')
            response_attr_dict[gid]['peak_index'] = np.array([peak_index], dtype='uint32')
            print('Rank %i: took %.2f s to append compute prediction attributes for %s gid %i' % \
                  (rank, time.time() - local_time, target_population, gid))
            count += 1
        if not debug:
            append_cell_attributes(comm, features_path, target_population, response_attr_dict,
                                   namespace=prediction_namespace, io_size=io_size, chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
        sys.stdout.flush()
        del response
        del response_attr_dict
        gc.collect()

    global_count = comm.gather(count, root=0)
    if rank == 0:
        print('%i ranks took %.2f s to compute selectivity parameters for %i %s cells' % \
              (comm.size, time.time() - start_time, np.sum(global_count), target_population))


if __name__ == '__main__':
    main(args=sys.argv[(list_find(lambda s: s.find(script_name) != -1,sys.argv)+1):])
