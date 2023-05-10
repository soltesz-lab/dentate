import os, sys, gc, logging, string, time, itertools, uuid
from mpi4py import MPI
import click
from collections import defaultdict
import numpy as np
import dentate
from dentate import cells, neuron_utils, synapses, utils, cell_clamp, io_utils
from dentate.env import Env
from dentate.neuron_utils import configure_hoc_env, load_cell_template
from dentate.utils import *
from neuroh5.io import NeuroH5TreeGen, append_cell_attributes, read_population_ranges, read_cell_attribute_selection, read_graph_selection
import h5py

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


# This logger will inherit its settings from the root logger, created in dentate.env
logger = get_module_logger(__name__)
            
        
@click.command()
@click.option("--config", '-c', required=True, type=str, help='model configuration file name')
@click.option("--config-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              default='config',
              help='path to directory containing network and cell mechanism config files')
@click.option("--input-path", required=False, type=click.Path(exists=True, file_okay=True, dir_okay=False), \
              help='optional path to data file if different than data files specified in config')
@click.option("--population", '-p', required=True, type=str, default='GC', help='target population')
@click.option("--template-paths", type=str, required=True,
              help='colon-separated list of paths to directories containing hoc cell templates')
@click.option("--dataset-prefix", required=True, type=click.Path(exists=True, file_okay=False, dir_okay=True),
              help='path to directory containing required neuroh5 data files')
@click.option("--results-path", required=False, type=click.Path(exists=True, file_okay=False, dir_okay=True), \
              help='path to directory where output files will be written')
@click.option("--results-file-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 files that contain output spike and intracellular trace data')
@click.option("--results-namespace-id", type=str, required=False, default=None, \
              help='identifier that is used to name neuroh5 namespaces that contain output spike and intracellular trace data')
@click.option("--v-init", type=float, default=-75.0, help='initialization membrane potential (mV)')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, config_prefix, input_path, population, template_paths, dataset_prefix, results_path, results_file_id, results_namespace_id, v_init, io_size, chunk_size, value_chunk_size, write_size, verbose):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)

    if io_size == -1:
        io_size = comm.size

    if results_file_id is None:
        if rank == 0:
            result_file_id = uuid.uuid4()
        results_file_id = comm.bcast(results_file_id, root=0)
    if results_namespace_id is None:
        results_namespace_id = 'Cell Clamp Results'
    comm = MPI.COMM_WORLD
    np.seterr(all='raise')
    verbose = True
    params = dict(locals())
    env = Env(**params)
    configure_hoc_env(env)
    if rank == 0:
        io_utils.mkout(env, env.results_file_path)
    env.comm.barrier()
    env.cell_selection = {}
    template_class = load_cell_template(env, population)

    if input_path is not None:
        env.data_file_path = input_path
        env.load_celltypes()
    
    synapse_config = env.celltypes[population]['synapses']

    weights_namespaces = []
    if 'weights' in synapse_config:
        has_weights = synapse_config['weights']
        if has_weights:
            if 'weights namespace' in synapse_config:
                weights_namespaces.append(synapse_config['weights namespace'])
            elif 'weights namespaces' in synapse_config:
                weights_namespaces.extend(synapse_config['weights namespaces'])
            else:
                weights_namespaces.append('Weights')
    else:
        has_weights = False
    
    start_time = time.time()
    count = 0
    gid_count = 0
    attr_dict = {}
    if input_path is None:
        cell_path = env.data_file_path
        connectivity_path = env.connectivity_file_path
    else:
        cell_path = input_path
        connectivity_path = input_path
        
    for gid, morph_dict in NeuroH5TreeGen(cell_path, population, io_size=io_size, comm=env.comm, topology=True):
        local_time = time.time()
        if gid is not None:
            color = 0
            comm0 = comm.Split(color, 0)

            logger.info('Rank %i gid: %i' % (rank, gid))
            cell_dict = { 'morph': morph_dict }
            synapses_iter = read_cell_attribute_selection(cell_path, population, [gid], 
                                                          'Synapse Attributes',
                                                          comm=comm0)
            _, synapse_dict = next(synapses_iter)
            cell_dict['synapse'] = synapse_dict
            
            if has_weights:
                cell_weights_iters = [read_cell_attribute_selection(cell_path, population, [gid],
                                                                    weights_namespace, comm=comm0)
                                          for weights_namespace in weights_namespaces]
                weight_dict = dict(zip_longest(weights_namespaces, cell_weights_iters))
                cell_dict['weight'] = weight_dict
                
            (graph, a) = read_graph_selection(file_name=connectivity_path, selection=[gid],
                                              namespaces=['Synapses', 'Connections'], comm=comm0)
            cell_dict['connectivity'] = (graph, a)
            
            gid_count += 1

            attr_dict[gid] = {}
            attr_dict[gid].update(cell_clamp.measure_passive(gid, population, v_init, env, cell_dict=cell_dict))
            attr_dict[gid].update(cell_clamp.measure_ap(gid, population, v_init, env, cell_dict=cell_dict))
            attr_dict[gid].update(cell_clamp.measure_ap_rate(gid, population, v_init, env, cell_dict=cell_dict))
            attr_dict[gid].update(cell_clamp.measure_fi(gid, population, v_init, env, cell_dict=cell_dict))

        else:
            color = 1
            comm0 = comm.Split(color, 0)
            logger.info('Rank %i gid is None' % (rank))
        comm0.Free()

        count += 1
        if (results_path is not None) and (count % write_size == 0):
            append_cell_attributes(env.results_file_path, population, attr_dict,
                                   namespace=env.results_namespace_id,
                                   comm=env.comm, io_size=env.io_size,
                                   chunk_size=chunk_size,
                                   value_chunk_size=value_chunk_size)
            attr_dict = {}
    
        
    env.comm.barrier()
    if results_path is not None:
        append_cell_attributes(env.results_file_path, population, attr_dict,
                               namespace=env.results_namespace_id,
                               comm=env.comm, io_size=env.io_size,
                               chunk_size=chunk_size,
                               value_chunk_size=value_chunk_size)
    global_count = env.comm.gather(gid_count, root=0)

    MPI.Finalize()


if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
