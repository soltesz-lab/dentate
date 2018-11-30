
import sys, os, time, gc, click, logging, yaml, pprint
from collections import defaultdict
import numpy as np
from mpi4py import MPI
from neuroh5.io import read_graph_selection, read_population_ranges
import h5py
from dentate.env import Env
import dentate.utils as utils
import dentate.synapses as synapses

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook

@click.command()
@click.option("--config", '-c', required=True, type=str)
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config', help='path to directory containing network config files')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
                help='path to directory containing required neuroh5 data files')
@click.option("--selection-path", type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--output-path", '-o', required=True, type=click.Path(exists=False, file_okay=True, dir_okay=False))
@click.option("--population", '-p', type=str, required=True)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, config_prefix, dataset_prefix, selection_path, output_path, population, io_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))

    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config, config_prefix=config_prefix, dataset_prefix=dataset_prefix)

    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    pop_ranges, pop_size = read_population_ranges(env.connectivity_file_path, comm=comm)

    count = 0
    gid_count = 0
    start_time = time.time()

    gid_range = None
    if selection_path is not None:
        gid_range = np.loadtxt(selection_path, dtype=np.uint32)

    postsyn_name = population.encode('ascii','ignore')
    presyn_names = env.projection_dict[population]
    selection_projections = [ (presyn_name, postsyn_name) for presyn_name in presyn_names ]
        
    (graph, a) = read_graph_selection(env.connectivity_file_path, 
                                      selection=gid_range.tolist(), comm=env.comm)


    output_dict = defaultdict(set)
    
    for presyn_name in presyn_names:
        edge_iter = graph[postsyn_name][presyn_name]
        for (postsyn_gid, edges) in edge_iter:
            presyn_gids, edge_attrs = edges
            output_dict[postsyn_name].add(int(postsyn_gid))
            output_dict[presyn_name].update(np.asarray(presyn_gids,dtype=np.intp).tolist())

    yaml_output_dict = {}
    for k, v in utils.viewitems(output_dict):
        yaml_output_dict[k] = list(v)
        
    with open(output_path, 'w') as outfile:
        yaml.dump(yaml_output_dict, outfile)

    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])

