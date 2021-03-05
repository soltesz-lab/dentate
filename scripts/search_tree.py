import itertools, logging, os, sys, time
from collections import defaultdict
from mpi4py import MPI
import h5py
import numpy as np
import click
import dentate.utils as utils
import dentate.cells as cells
import dentate.synapses as synapses
from dentate.env import Env
from dentate.neuron_utils import configure_hoc_env, load_cell_template
from neuroh5.io import NeuroH5TreeGen, scatter_read_trees, read_population_ranges, read_tree_selection
from neuron import h

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
#sys.excepthook = mpi_excepthook

script_name=os.path.basename(__file__)

def get_distance_to_node(cell, source_sec, target_sec, loc=0.5):
    return h.distance(source_sec(0.5), target_sec(loc))

def compare_points(xyz1, xyz2):
    if xyz1[0].shape[0] == xyz2[0].shape[0]:
        print(f'xyz1 = {xyz1[0][:10]} xyz2 = {xyz2[0][:10]}') 
        print(f'xyz1 = {xyz1[1][:10]} xyz2 = {xyz2[1][:10]}') 
        sys.stdout.flush()
        return all(map(lambda ab: np.all(np.isclose(ab[0], ab[1], atol=1e-4, rtol=1e-4)), zip(xyz1, xyz2)))
    else:
        return False

@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str)
@click.option("--prototype-gid", type=int, default=0)
@click.option("--prototype-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--population", required=True, type=str)
@click.option("--io-size", type=int, default=-1)
@click.option("--verbose", "-v", is_flag=True)
def main(config, template_path, prototype_gid, prototype_path, forest_path, population, io_size, verbose):
    """

    :param config:
    :param template_path:
    :param prototype_gid:
    :param prototype_path:
    :param forest_path:
    :param population:
    :param io_size:
    """

    utils.config_logging(verbose)
    logger = utils.get_script_logger(script_name)
        
    comm = MPI.COMM_WORLD
    rank = comm.rank
    
    env = Env(comm=MPI.COMM_WORLD, config_file=config, template_paths=template_path)
    configure_hoc_env(env)
    
    if io_size == -1:
        io_size = comm.size
    if rank == 0:
        logger.info('%i ranks have been allocated' % comm.size)
    
    layers = env.layers
    layer_idx_dict = { layers[layer_name]: layer_name 
                       for layer_name in ['GCL', 'IML', 'MML', 'OML', 'Hilus'] }

    (tree_iter, _) = read_tree_selection(prototype_path, population, selection=[prototype_gid])
    (_, prototype_morph_dict) = next(tree_iter)
    prototype_x = prototype_morph_dict['x']
    prototype_y = prototype_morph_dict['y']
    prototype_z = prototype_morph_dict['z']
    prototype_xyz = (prototype_x, prototype_y, prototype_z)

    (pop_ranges, _) = read_population_ranges(forest_path, comm=comm)
    start_time = time.time()

    (population_start, _) = pop_ranges[population]
    template_class = load_cell_template(env, population, bcast_template=True)
    for gid, morph_dict in NeuroH5TreeGen(forest_path, population, io_size=io_size, cache_size=1, comm=comm, topology=True):
#    trees, _ = scatter_read_trees(forest_path, population, io_size=io_size, comm=comm, topology=True)
 #   for gid, morph_dict in trees:
        if gid is not None:
            logger.info('Rank %i gid: %i' % (rank, gid))
            secnodes_dict = morph_dict['section_topology']['nodes']
            vx = morph_dict['x']
            vy = morph_dict['y']
            vz = morph_dict['z']
            if compare_points((vx,vy,vz), prototype_xyz):
                logger.info('Possible match: gid %i' % gid)
    MPI.Finalize()



if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
