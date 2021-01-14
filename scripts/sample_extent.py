import sys, os, gc, pprint, time, click, pprint
from collections import defaultdict
from mpi4py import MPI
import numpy as np
import yaml
import dentate
from dentate import utils, io_utils
from dentate.env import Env
from neuroh5.io import read_cell_attributes, read_graph_selection, read_population_ranges
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

def euclidean_distance(a, b):
    """Row-wise euclidean distance.
    a, b are row vectors of points.
    """
    return np.sqrt(np.sum((a - b) ** 2, axis=1))

@click.command()
@click.option("--arena-id", required=False, type=str,
              help='name of arena used for spatial stimulus')
@click.option("--bin-sample-count", type=int, required=False, help='Number of samples per spatial bin')
@click.option("--bin-sample-proximal-pf", is_flag=True, required=False, help='Only sample with a place field overlapping the origin')
@click.option("--config", '-c', required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config', help='path to directory containing network config files')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
                help='path to directory containing required neuroh5 data files')
@click.option("--distance-bin-extent", type=float, default=1000., help='Longitudinal extent of sample bin in micrometers')
@click.option("--distances-namespace", '-n', type=str, default='Arc Distances')
@click.option("--input-features-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--input-features-namespaces", type=str, multiple=True, default=['Place Selectivity', 'Grid Selectivity'])
@click.option("--populations", '-i', required=True, multiple=True, type=str)
@click.option("--spike-input-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False),
                  help='path to file for input spikes when cell selection is specified')
@click.option("--spike-input-namespace", required=False, type=str,
                  help='namespace for input spikes when cell selection is specified')
@click.option("--spike-input-attr", required=False, type=str,
                  help='attribute name for input spikes when cell selection is specified')
@click.option("--output-path", '-o', required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True))
@click.option("--io-size", type=int, default=-1)
@click.option("--trajectory-id", required=False, type=str,
              help='name of trajectory used for spatial stimulus')
@click.option("--write-selection", is_flag=True)
@click.option("--verbose", "-v", is_flag=True)
def main(arena_id, bin_sample_count, bin_sample_proximal_pf, config, config_prefix, dataset_prefix, distances_namespace, distance_bin_extent, input_features_path, input_features_namespaces, populations, spike_input_path, spike_input_namespace, spike_input_attr, output_path, io_size, trajectory_id, write_selection, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    if (bin_sample_count is None) and (bin_sample_proximal_pf is None):
        raise RuntimeError("Neither --bin-sample-count nor --bin-sample-proximal-pf is specified.")

    if (bin_sample_proximal_pf is not None) and (input_features_path is None):
        raise RuntimeError("--input-features-path must be specified when --bin-sample-proximal-pf is given.")

    
    env = Env(comm=comm, config_file=config, 
              config_prefix=config_prefix, dataset_prefix=dataset_prefix, 
              results_path=output_path, spike_input_path=spike_input_path, 
              spike_input_namespace=spike_input_namespace, spike_input_attr=spike_input_attr,
              arena_id=arena_id, trajectory_id=trajectory_id)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    pop_ranges, pop_size = read_population_ranges(env.connectivity_file_path, comm=comm)

    distance_U_dict = {}
    distance_V_dict = {}
    range_U_dict = {}
    range_V_dict = {}

    selection_dict = defaultdict(set)

    comm0 = env.comm.Split(2 if rank == 0 else 0, 0)

    local_random = np.random.RandomState()
    local_random.seed(1000)

    if len(populations) == 0:
        populations = sorted(pop_ranges.keys())
    
    
    if rank == 0:
        for population in populations:
            distances = read_cell_attributes(env.data_file_path, population, namespace=distances_namespace, comm=comm0)

            soma_distances = {}
            num_fields_dict = {}
            field_xy_dict = {}
            field_width_dict = {}
            if input_features_path is not None:
                for input_features_namespace in input_features_namespaces:
                    if arena_id is not None:
                        this_features_namespace = '%s %s' % (input_features_namespace, arena_id)
                    else:
                        this_features_namespace = input_features_namespace
                    input_features_iter = read_cell_attributes(input_features_path, population, 
                                                               namespace=this_features_namespace,
                                                               mask=set(['Num Fields', 'Field Width',
                                                                         'X Offset', 'Y Offset']), 
                                                               comm=comm0)
                    count = 0
                    for gid, attr_dict in input_features_iter:
                        num_fields_dict[gid] = attr_dict['Num Fields']
                        field_width_dict[gid] = attr_dict['Field Width']
                        field_xy_dict[gid] = (attr_dict['X Offset'], attr_dict['Y Offset'])
                        count += 1
                    logger.info('Read feature data from namespace %s for %i cells in population %s' % (this_features_namespace, count, population))

                for (gid, v) in distances:
                    num_fields = num_fields_dict.get(gid, 0)
                    if num_fields > 0:
                        soma_distances[gid] = (v['U Distance'][0], v['V Distance'][0])
            else:
                for (gid, v) in distances:
                    soma_distances[gid] = (v['U Distance'][0], v['V Distance'][0])
            
            numitems = len(list(soma_distances.keys()))
            logger.info('read %s distances (%i elements)' % (population, numitems))

            if numitems == 0:
                continue

            gid_array = np.asarray([gid for gid in soma_distances])
            distance_U_array = np.asarray([soma_distances[gid][0] for gid in gid_array])
            distance_V_array = np.asarray([soma_distances[gid][1] for gid in gid_array])

            U_min = np.min(distance_U_array)
            U_max = np.max(distance_U_array)
            V_min = np.min(distance_V_array)
            V_max = np.max(distance_V_array)

            range_U_dict[population] = (U_min, U_max)
            range_V_dict[population] = (V_min, V_max)
            
            distance_U = { gid: soma_distances[gid][0] for gid in soma_distances }
            distance_V = { gid: soma_distances[gid][1] for gid in soma_distances }
            
            distance_U_dict[population] = distance_U
            distance_V_dict[population] = distance_V
            
            min_dist = U_min
            max_dist = U_max 

            distance_bins = np.arange(U_min, U_max, distance_bin_extent)
            distance_bin_array = np.digitize(distance_U_array, distance_bins)

            selection_set = set([])
            for bin_index in range(len(distance_bins)+1):
                bin_gids = gid_array[np.where(distance_bin_array == bin_index)[0]]
                if len(bin_gids) > 0:
                    if bin_sample_proximal_pf:
                        proximal_bin_gids = []
                        for gid in bin_gids:
                            x, y = field_xy_dict[gid]
                            fw = field_width_dict[gid]
                            dist_origin = euclidean_distance(np.column_stack((x,y)), np.asarray([0., 0.]))
                            if np.any(dist_origin < fw):
                               proximal_bin_gids.append(gid)
                        proximal_bin_gids = np.asarray(proximal_bin_gids)
                        selected_bin_gids = local_random.choice(proximal_bin_gids, replace=False, size=bin_sample_count)
                    elif bin_sample_count:
                        selected_bin_gids = local_random.choice(bin_gids, replace=False, size=bin_sample_count)
                    for gid in selected_bin_gids:
                        selection_set.add(int(gid))
            logger.info('selected %i cells from population %s' % (len(selection_set), population))
            selection_dict[population] = selection_set

        yaml_output_dict = {}
        for k, v in utils.viewitems(selection_dict):
            yaml_output_dict[k] = list(sorted(v))
         
        yaml_output_path = '%s/DG_slice.yaml' % output_path
        with open(yaml_output_path, 'w') as outfile:
            yaml.dump(yaml_output_dict, outfile)

        del(yaml_output_dict)

    env.comm.barrier()

    write_selection_file_path = None
    if write_selection:
        write_selection_file_path =  "%s/%s_selection.h5" % (env.results_path, env.modelName)

    if write_selection_file_path is not None:
        if rank == 0:
            io_utils.mkout(env, write_selection_file_path)
        env.comm.barrier()
        selection_dict = env.comm.bcast(dict(selection_dict), root=0)
        env.cell_selection = selection_dict
        io_utils.write_cell_selection(env, write_selection_file_path, populations=populations)
        input_selection = io_utils.write_connection_selection(env, write_selection_file_path,
                                                              populations=populations)

        if env.spike_input_ns is not None:
            io_utils.write_input_cell_selection(env, input_selection, write_selection_file_path,
                                                populations=populations)
    env.comm.barrier()
    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
