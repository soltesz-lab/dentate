##
## Generates distance-weighted random connectivity between the specified populations.
##

import sys, os, os.path, gc, click, logging
from mpi4py import MPI
from neuroh5.io import read_population_ranges, read_population_names, bcast_cell_attributes, read_cell_attributes
from neuron import h
import h5py
import numpy as np
import rbf
import dentate
from dentate.gapjunctions import generate_gj_connections
from dentate.env import Env
import dentate.utils as utils
from dentate.neuron_utils import configure_hoc_env

sys_excepthook = sys.excepthook
def mpi_excepthook(type, value, traceback):
    sys_excepthook(type, value, traceback)
    if MPI.COMM_WORLD.size > 1:
        MPI.COMM_WORLD.Abort(1)
sys.excepthook = mpi_excepthook


@click.command()
@click.option("--config", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--template-path", type=str, default='templates')
@click.option("--types-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--forest-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--connectivity-path", required=True, type=click.Path())
@click.option("--connectivity-namespace", type=str, default='Gap Junctions')
@click.option("--coords-path", required=True, type=click.Path(exists=True, file_okay=True, dir_okay=False))
@click.option("--coords-namespace", type=str, default='Coordinates')
@click.option("--io-size", type=int, default=-1)
@click.option("--chunk-size", type=int, default=1000)
@click.option("--value-chunk-size", type=int, default=1000)
@click.option("--cache-size", type=int, default=50)
@click.option("--write-size", type=int, default=1)
@click.option("--verbose", "-v", is_flag=True)
@click.option("--dry-run", is_flag=True)
def main(config, template_path, types_path, forest_path, connectivity_path, connectivity_namespace, coords_path, coords_namespace,
         io_size, chunk_size, value_chunk_size, cache_size, write_size, verbose, dry_run):

    utils.config_logging(verbose)
    logger = utils.get_script_logger(os.path.basename(__file__))
    
    comm = MPI.COMM_WORLD
    rank = comm.rank

    env = Env(comm=comm, config_file=config, template_paths=template_path)
    configure_hoc_env(env)

    gj_config = env.gapjunctions
    gj_seed = int(env.modelConfig['Random Seeds']['Gap Junctions'])

    soma_coords = {}

    if (not dry_run) and (rank==0):
        if not os.path.isfile(connectivity_path):
            input_file  = h5py.File(types_path,'r')
            output_file = h5py.File(connectivity_path,'w')
            input_file.copy('/H5Types',output_file)
            input_file.close()
            output_file.close()
    comm.barrier()
    
    population_ranges = read_population_ranges(coords_path)[0]
    populations = list(population_ranges.keys())
    
    if rank == 0:
        logger.info('Reading population coordinates...')

    soma_distances = {}
    for population in populations:
        coords_iter = bcast_cell_attributes(coords_path, population, 0, namespace=coords_namespace)

        soma_coords[population] = { k: (v['X Coordinate'][0], v['Y Coordinate'][0], v['Z Coordinate'][0]) for (k,v) in coords_iter }

        gc.collect()

    generate_gj_connections(env, forest_path, soma_coords, gj_config, gj_seed, connectivity_namespace, connectivity_path,
                            io_size, chunk_size, value_chunk_size, cache_size,
                            dry_run=dry_run)

        
    MPI.Finalize()

if __name__ == '__main__':
    main(args=sys.argv[(utils.list_find(lambda x: os.path.basename(x) == os.path.basename(__file__), sys.argv)+1):])
